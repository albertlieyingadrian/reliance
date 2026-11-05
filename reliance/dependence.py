"""Attribution methods.

Every function takes (g, x, v) and returns one number per position: `g` a callable x -> (n,),
`x` an activation (n, p, c) as `layout.positions` returns it, `v` a unit direction. The gradient is written `dg`, since `g` is
the readout. Backward measures differentiate g, forward ones only evaluate it.
Closed forms for all of them are in tests/test_dependence.py.
"""
import torch
from einops import rearrange


def gradient(g, x, create_graph=True):
    """dg/dx, the same shape as x. `create_graph` keeps it differentiable, which a penalty needs."""
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)

    dg, = torch.autograd.grad(g(x).sum(), x, create_graph=create_graph)

    return dg


def cone_directions(v, spread, samples):
    """`samples` unit directions in a cap of half-angle about arctan(spread) around v.

    The jitter is divided by sqrt(c) so `spread` means a fraction of ||v|| rather than growing
    with the width: an undivided gaussian has norm about sqrt(c).
    """
    drift = torch.randn(samples, *v.shape, device=v.device) / v.shape[-1] ** 0.5
    u = v + spread * v.norm() * drift

    return u / u.norm(dim=-1, keepdim=True)


def directional_derivative(g, x, v, cone=0.0, samples=10, create_graph=True):
    """<dg, v>: the directional derivative along v. On a linear readout, <w, v> everywhere.

    `cone` spreads v over a cap of that size, for an estimate you do not trust to the degree.
    A flag rather than a job for `cone_average` because <dg, u> is linear in u: one gradient
    serves every direction, so the cone costs what one reading costs.
    """
    dg = gradient(g, x, create_graph)
    found = dg @ v if not cone else (dg @ cone_directions(v, cone, samples).T).mean(-1)

    return found


def gradinput(g, x, v, create_graph=True):
    """<dg, v> <x, v>: the slope times how far along v this example sits. First-order contribution."""
    dg = gradient(g, x, create_graph)
    found = (dg @ v) * (x @ v)

    return found


def integrated(g, x, v, steps=8, create_graph=True):
    """gradinput averaged along the straight line from an empty activation to this one.

    Midpoint rule, so no step sits at alpha = 0 where the readout may be undefined.
    """
    total = 0.0

    for step in range(steps):
        alpha = (step + 0.5) / steps
        total = total + gradient(g, alpha * x, create_graph)

    dg = total / steps
    found = (dg @ v) * (x @ v)

    return found


def smoothgrad(g, x, v, samples=16, delta=0.2, cone=0.0, create_graph=True):
    """The directional derivative averaged over a ball of radius delta ||x||, per position.

    Relative to the activation's own norm, so the radius asks the same question at any depth.
    """
    scale = delta * x.detach().norm(dim=-1, keepdim=True)
    total = 0.0

    for _ in range(samples):
        total = total + gradient(g, x + scale * torch.randn_like(x), create_graph)

    dg = total / samples
    found = dg @ v if not cone else (dg @ cone_directions(v, cone, 10).T).mean(-1)

    return found


def finite_difference(g, x, v, delta=0.2):
    """g(x + h v / 2) - g(x - h v / 2) with h = delta ||x||. Two forward passes, no backward.

    Shape (n, 1): the shift moves every position at once, so this has no position resolution and
    broadcasts against the measures that do.
    """
    step = (delta / 2) * x.detach().norm(dim=-1, keepdim=True) * v

    return rearrange(g(x + step) - g(x - step), 'n -> n 1')


def project_out(x, v):
    """x with its component along v removed -- or, for orthonormal rows V (k, c), its component in
    their span. The same projection whether x is (n, p, c) or (n, c)."""
    V = v[None] if v.ndim == 1 else v

    return x - (x @ V.T) @ V


def banzhaf(g, x, v):
    """g(x) - g(x with v projected out): what the answer loses when the direction goes.

    Two forward passes, no gradient. How the gaps are then aggregated -- absolute value or square
    -- is the penalty's business, not this function's. Handed a basis (k, c) it removes the whole
    span at once, so atoms the readout uses with opposite signs cancel and cost nothing; read one
    atom at a time through `subspace_map` they are two violations. Its first-order expansion over
    a span is `gradinput` summed over the atoms.
    """
    return rearrange(g(x) - g(project_out(x, v)), 'n -> n 1')


def forward_sensitivity(g, x, v, samples=16, delta=0.2):
    """E[(g(x + eps v) - g(x))^2] with eps ~ N(0, (delta ||x||)^2). One per example, never negative.

    Not an attribution. Where `directional_derivative` asks the answer to be flat along v in a vanishing ball,
    this asks it to be unchanged under a push of a real size -- the demand actually being made.

    """
    base = g(x)
    scale = delta * x.detach().norm(dim=-1, keepdim=True)
    total = 0.0

    for _ in range(samples):
        eps = scale * rearrange(torch.randn(len(x), device=x.device), 'n -> n 1 1')
        total = total + (g(x + eps * v) - base).pow(2)

    return rearrange(total / samples, 'n -> n 1')


def cone_average(measure, g, x, v, spread=0.2, samples=10, **opts):
    """`measure` averaged over a cone of directions around v.

    Every measure here treats v as exact, and v is never exact -- a difference of means on a few
    hundred examples has its own error bar, and the direction it returns is a draw around the one
    that was wanted. This asks the question that survives that: jitter v by `spread` of its own
    norm, renormalise, average over the cone. The perturbation is divided by sqrt(c) so that `spread` means a fraction of ||v||
    rather than growing with the width -- an undivided gaussian has norm about sqrt(c) -- which
    puts the samples in a spherical cap of half-angle about arctan(spread), so the default 0.2 is
    roughly eleven degrees.

    It buys steadiness and pays in sharpness: a v that merely lies near a direction the readout
    uses will now report some dependence, because part of the cone reaches that direction. That is
    the trade, and it is the right one exactly when v is not to be trusted to the last degree.

    Where it does nothing. `directional_derivative` and `smoothgrad` are both exactly linear in
    v -- the gradient is taken at x and never sees v, so <dg, u> is linear in u however nonlinear
    the readout is -- and averaging a linear function over the cone gives <dg, mean u>, which by
    symmetry is the point value times E[cos]. That is a shrinkage by 1 / sqrt(1 + spread^2) and no
    robustness at all.

    Where it does harm. `forward_sensitivity` is a squared quantity, so averaging non-negative
    values over a neighbourhood has nothing to cancel: the cone can only inflate it, and it lands
    about twice as far from the truth as reading it at v would have.

    So it helps where the measure is nonlinear in v *and* signed -- `gradinput`, `integrated`,
    `finite_difference`, `banzhaf` -- and nowhere else.
    """
    total = 0.0

    for u in cone_directions(v, spread, samples):
        total = total + measure(g, x, u, **opts)

    return total / samples


def baseline_directional_derivative(dg, width):
    """E over random unit u of <dg, u>^2 = ||dg||^2 / c, since E[u u^T] = I / c.

    Serves `directional_derivative` and `smoothgrad`, whose value is linear in v.
    """
    return dg.pow(2).sum(-1) / width


def baseline_gradinput(dg, x, width):
    """E over random unit u of (<dg, u> <x, u>)^2.

    The sphere's fourth moment gives (||dg||^2 ||x||^2 + 2 <dg, x>^2) / (c (c + 2)). Serves
    `gradinput` and `integrated`. Homogeneous in x, so the share is scale-invariant.
    """
    pair = (dg * x).sum(-1)

    return (dg.pow(2).sum(-1) * x.pow(2).sum(-1) + 2 * pair.pow(2)) / (width * (width + 2))


def make_measure_relative(measure, g, x, v, draws=16, **opts):
    """`measure` at v, in units of what it gives at a direction aimed at nothing.

    Squaring the result gives the share: about 1 at a random direction, larger where the readout
    leans on v. That is what makes a constraint nameable -- "a hundredth of chance" is a sentence
    about the model, where "P <= 1e-7" is a sentence about nobody.

    Works for every measure because it asks each one the same question, pointed nowhere. The price
    is `draws` extra evaluations, and it is a real price: at 4 draws the estimate is noisy enough
    to cost a point or two of accuracy downstream, and it settles by about 16. Named `draws` and
    not `samples` because several measures already have a `samples` of their own.
    """
    level = 0.0

    for _ in range(draws):
        u = torch.randn_like(v)
        level = level + measure(g, x, u / u.norm(), **opts).pow(2)

    return measure(g, x, v, **opts) / (level / draws).clamp(min=1e-30).sqrt()


def subspace_map(measure, g, x, V, **opts):
    """Run a measure once per atom of an orthonormal subspace V (k, c). -> (n, p, k).

    A loop, not a batched form: every atom is a violation in its own right, and summing them
    first would let one atom's positive value cancel another's.
    """
    return rearrange(torch.stack([measure(g, x, v, **opts) for v in V]), 'k n p -> n p k')
