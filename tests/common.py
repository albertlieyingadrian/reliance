"""Training a readout against a method, and finding out what its lambda has to be.

A method wins if some strength gets counterfactual accuracy past PASSING. `at` is the strength
that won; across regimes that column says whether lambda is a constant or a conversion factor.
`ceiling` is project_out, which nothing can beat, and says whether a regime is winnable at all.
"""
import torch
import torch.nn.functional as F

from reliance import dependence as D
from reliance import intervene as I
from reliance.lagrangian import DualRate

STRENGTHS = [10.0 ** e for e in range(-3, 4)]
PASSING = 0.9

MEASURES = ('directional_derivative', 'gradinput', 'integrated', 'smoothgrad',
            'finite_difference', 'banzhaf', 'forward_sensitivity')
NON_NEGATIVE = ('forward_sensitivity',)
INTERVENTIONS = ('pps', 'project_out', 'noise', 'shuffle')
NO_STRENGTH = ('project_out', 'shuffle')
TRAIN_ONLY = ('pps', 'noise', 'shuffle')


class Net(torch.nn.Module):
    """The readout being defended. GELU at every depth, so the depth axis varies only depth.

    It was Tanh at depth 1 and GELU deeper, which meant a depth-1 to depth-3 comparison changed
    the nonlinearity at the same time and neither could be blamed. GELU also saturates less on one
    side, which is the failure that made scale 100 unwinnable.
    """

    def __init__(self, d, hidden=32, depth=1):
        super().__init__()
        layers, width = [], d

        for _ in range(depth):
            layers += [torch.nn.Linear(width, hidden), torch.nn.GELU()]
            width = hidden

        self.body = torch.nn.Sequential(*layers, torch.nn.Linear(width, 1))

    def forward(self, x):
        return self.body(x).squeeze(-1).squeeze(-1)


def spread(dgp, n=2048):
    """std of <x, v>: the unit a finite perturbation would be quoted in, if one still were."""
    x, _ = dgp.sample(n)

    return float((x @ dgp.get_spurious()).std())


def penalty_of(name, power=2.0, relative=False, draws=16):
    """A measure as a scalar penalty: |value|^power, averaged.

    `power` 2 weights a few large violations far above many small ones; 1 is the absolute value and
    weights them evenly, which is the usual reason to reach for an L1. Below 1 the derivative at
    zero is infinite and the value passes through zero constantly, so |A|^p is smoothed inside the
    power -- the Charbonnier form -- which agrees with it wherever A is bigger than the epsilon.

    `relative` wraps the measure in `dependence.make_measure_relative`, so the penalty reads as a multiple of what
    a direction aimed at nothing collects and a target can be named as a fraction of chance.
    """
    measure = getattr(D, name)
    options = {'smoothgrad': {'samples': 4}, 'integrated': {'steps': 4},
               'forward_sensitivity': {'samples': 4}}.get(name, {})


    def penalty(g, x, v):
        found = (D.make_measure_relative(measure, g, x, v, draws=draws, **options) if relative
                 else measure(g, x, v, **options))

        if name in NON_NEGATIVE:
            return found.pow(power / 2).mean() if power != 2 else found.mean()

        if power == 2:
            return found.pow(2).mean()

        smooth = 1e-6 * found.detach().pow(2).mean().clamp(min=1e-30)

        return (found.pow(2) + smooth).pow(power / 2).mean()

    return penalty


def intervention_of(name):
    """An intervention as `apply(x, v, strength)`, with strength a fraction of the activation scale.

    The methods take an absolute strength, as the paper does. Making it portable is the harness's
    job, and it is done with the batch's *mean* norm rather than each example's: `pps` has to move
    every example by the same amount or it stops being a constant shift, which is the property the
    whole argument about it rests on.
    """
    method = getattr(I, name)

    def apply(x, v, strength):
        if name in NO_STRENGTH:
            return method(x, v)

        return method(x, v, strength * float(x.norm(dim=-1).mean()))

    return apply


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def fit(dgp, penalty=None, lam=0.0, dual=None, intervene=None, strength=0.0,
        n=1024, steps=400, lr=0.01, depth=1, device=None):
    """Train once. A penalty adds to the loss; an intervention rewrites what the net sees.

    The device lives here and not in the generator: the toy is a handful of matmuls and where they
    run is a property of the harness, not of the problem.
    """
    device = DEVICE if device is None else device
    x, y = dgp.sample(n)
    x, y = x.to(device), y.to(device)
    v = dgp.get_spurious().to(device)
    net = Net(dgp.d, depth=depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    params = list(net.parameters())

    if dual is not None and not hasattr(dual, 'step'):
        dual = dual(net, x, v)        # a controller whose target closes over the net and the data

    for _ in range(steps):
        seen = x if intervene is None else intervene(x, v, strength)
        task = F.binary_cross_entropy_with_logits(net(seen), y)

        if penalty is None or not (lam or dual is not None):
            opt.zero_grad()
            task.backward()
            opt.step()
            continue

        found = penalty(net, x, v)

        if dual is not None:
            opt.zero_grad()
            dual.step(task, found, params, opt)
            continue

        opt.zero_grad()
        (task + lam * found).backward()
        opt.step()

    return net


def scores(net, dgp, n=4000, on_test=None):
    """Counterfactual accuracy: does the model follow the core when the two rules disagree."""
    device = next(net.parameters()).device
    x, y = dgp.counterfactual(n)
    x, y = x.to(device), y.to(device)

    if on_test is not None:
        x = on_test(x, dgp.get_spurious().to(device), 0.0)

    with torch.no_grad():
        return float(((net(x) > 0).float() == y).float().mean())


def run(name, dgp, strength, **opts):
    """One method at one strength. Penalties and interventions look alike from out here."""
    if name in INTERVENTIONS:
        apply = intervention_of(name)
        net = fit(dgp, intervene=apply, strength=strength, **opts)

        return scores(net, dgp, on_test=None if name in TRAIN_ONLY else apply)

    return scores(fit(dgp, penalty_of(name), lam=strength, **opts), dgp)


def scan(name, dgp, strengths=STRENGTHS, **opts):
    return {s: run(name, dgp, s, **opts) for s in strengths}


def verdict(name, dgp, strengths=STRENGTHS, passing=PASSING, **opts):
    """Does this method win, how well, and over how wide a range of strength."""
    got = scan(name, dgp, strengths, **opts)
    clears = [s for s, acc in got.items() if acc >= passing]

    return {'best': max(got.values()), 'at': max(got, key=got.get), 'span': len(clears),
            'wins': bool(clears), 'by_strength': got}


def baseline(dgp, **opts):
    """No method at all. Near chance: the two rules disagree on every counterfactual row."""
    return scores(fit(dgp, **opts), dgp)


def ceiling(dgp, **opts):
    """project_out at train and test. Nothing can beat it, so it bounds every method here."""
    return run('project_out', dgp, 0.0, **opts)


def by_dual_rate(name, dgp, target=0.01, penalty=None, **opts):
    """lambda from the gradients, kappa from a constraint named against chance.

    The penalty is built `relative`, so it already reads as a multiple of what a random direction
    collects and the target is just a number. No sampling, nothing recomputed per step.
    """
    rate = DualRate(target=target)
    net = fit(dgp, penalty or penalty_of(name, relative=True), dual=rate, **opts)
    lams = [lam for lam, _, _ in rate.history]

    return {'accuracy': scores(net, dgp), 'peak': max(lams),
            'median': sorted(lams)[len(lams) // 2], 'satisfied': rate.satisfied(0.5)}
