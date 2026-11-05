"""Every measure has a closed form on a linear readout. This checks all of them against it.

For g(x) = sum_p <w, x_p> the gradient is w everywhere, so the seven measures collapse onto three
shapes: <w, v>, <w, v> <x_p, v>, and something proportional to sum_p ||x_p||. The agreements
between measures that are not obviously the same measure are the point, and each is a real claim.
"""
import pytest
import torch

from reliance import dependence as D


N, P, C = 6, 8, 5

PER_POSITION = ('directional_derivative', 'gradinput', 'integrated', 'smoothgrad')
PER_EXAMPLE = ('finite_difference', 'banzhaf', 'forward_sensitivity')
EVERY = PER_POSITION + PER_EXAMPLE


def measure(name):
    return getattr(D, name)


def problem(seed=0, orthogonal=False, positions=P):
    """A linear readout with a known weight, an activation, and a direction.

    `orthogonal` builds a readout that provably does not use v, by projecting v out of w.
    """
    torch.manual_seed(seed)

    x = torch.randn(N, positions, C)
    w = torch.randn(C)
    v = torch.randn(C)
    v = v / v.norm()

    if orthogonal:
        w = w - (w @ v) * v

    def g(z):
        return (z @ w).sum(1)

    return g, x, v, w


def close(got, want, tol=1e-5):
    return torch.allclose(got, torch.as_tensor(want).expand_as(got), atol=tol, rtol=tol)


def test_gradient_is_the_weight():
    g, x, v, w = problem()

    assert close(D.gradient(g, x), w.expand(N, P, C))


def test_directional_derivative_is_the_weight_along_v():
    g, x, v, w = problem()

    assert close(D.directional_derivative(g, x, v), (w @ v).expand(N, P))


def test_gradinput_is_the_weight_times_the_coordinate():
    g, x, v, w = problem()

    assert close(D.gradinput(g, x, v), (w @ v) * (x @ v))


def test_integrated_collapses_onto_gradinput():
    g, x, v, _ = problem()

    assert close(D.integrated(g, x, v, steps=3), D.gradinput(g, x, v))
    assert close(D.integrated(g, x, v, steps=64), D.gradinput(g, x, v))


def test_smoothgrad_collapses_onto_the_directional_derivative_with_no_sampling_error():
    g, x, v, w = problem()

    assert close(D.smoothgrad(g, x, v, samples=2), (w @ v).expand(N, P))
    assert close(D.smoothgrad(g, x, v, samples=2, delta=100.0), (w @ v).expand(N, P))


def test_smoothgrad_scales_its_ball_by_the_activation_norm():
    """delta is a fraction of ||x||, so the ball grows with what it sits on. A linear readout
    cannot witness that -- its gradient is the same everywhere and the noise averages out -- so
    this asks a quadratic one, where a single draw wobbles by exactly the size of its own ball.
    """
    torch.manual_seed(8)
    w = torch.randn(C)
    v = w / w.norm()

    def g(z):
        return (z @ w).pow(2).sum(1)

    x = torch.randn(N, P, C)

    def wobble(z):
        torch.manual_seed(9)
        draws = torch.stack([D.smoothgrad(g, z, v, samples=1) for _ in range(64)])

        return float(draws.detach().std(0).mean())

    assert abs(wobble(100 * x) / wobble(x) - 100.0) < 1.0


def test_finite_difference_is_delta_times_the_slope_times_the_activation_norm():
    """h = delta ||x_p||, so the swap is a fixed fraction of the activation rather than a fixed
    distance -- which is what makes one delta mean the same thing at any depth."""
    g, x, v, w = problem()

    for delta in (0.05, 0.2, 1.0):
        want = delta * (w @ v) * x.norm(dim=-1).sum(1)

        assert close(D.finite_difference(g, x, v, delta), want[:, None])


def test_forward_sensitivity_is_the_squared_slope_times_the_ball():
    torch.manual_seed(1)
    g, x, v, w = problem()

    for delta in (0.1, 0.5):
        want = (delta * x.norm(dim=-1).sum(1)) ** 2 * (w @ v) ** 2
        got = D.forward_sensitivity(g, x, v, samples=4000, delta=delta)

        assert close(got, want[:, None], tol=0.06)


def test_banzhaf_is_exact_when_there_is_one_position():
    """Nothing to split, so no sampling: the score with v, minus the score without it."""
    g, x, v, _ = problem(positions=1)

    once = D.banzhaf(g, x, v)
    twice = D.banzhaf(g, x, v)

    assert torch.equal(once, twice)
    assert close(once, D.gradinput(g, x, v))


def test_project_out_removes_exactly_the_component_along_v():
    g, x, v, _ = problem()

    assert close((D.project_out(x, v) @ v), torch.zeros(N, P), tol=1e-6)


@pytest.mark.parametrize('name', EVERY)
def test_a_readout_that_ignores_v_finds_no_dependence_on_it(name):
    """The one property every measure must have, or it is not measuring what it claims."""
    torch.manual_seed(5)
    g, x, v, _ = problem(orthogonal=True)

    assert measure(name)(g, x, v).abs().max() < 1e-4


@pytest.mark.parametrize('name', PER_POSITION)
def test_per_position_measures_return_one_number_per_position(name):
    g, x, v, _ = problem()

    assert measure(name)(g, x, v).shape == (N, P)


@pytest.mark.parametrize('name', PER_EXAMPLE)
def test_whole_activation_measures_return_one_number_per_example(name):
    g, x, v, _ = problem()

    assert measure(name)(g, x, v).shape == (N, 1)


@pytest.mark.parametrize('name', ('directional_derivative', 'gradinput', 'integrated'))
def test_cone_average_reduces_to_the_measure_at_zero_spread(name):
    """No cone, no averaging: it has to be the point measure exactly, or the smoothing is not
    something you can dial away. Only the measures that are themselves deterministic can say this
    -- `banzhaf` draws its own masks above one position, and would differ from itself.
    """
    g, x, v, _ = problem()

    assert close(D.cone_average(measure(name), g, x, v, spread=0.0, samples=3),
                 measure(name)(g, x, v))


def test_cone_average_samples_a_cap_of_the_stated_angle():
    """The geometry, read off a linear readout. There <dg, u> is linear in u, so averaging over
    the cone gives <w, mean u> = <w, v> E[cos], and for a gaussian jitter of `spread` of the norm
    E[cos] is 1 / sqrt(1 + spread^2). A cap of half-angle arctan(spread), as documented.
    """
    import math

    g, x, v, w = problem()
    point = (w @ v)

    for spread in (0.1, 0.2, 0.5):
        got = D.cone_average(D.directional_derivative, g, x, v, spread=spread, samples=4000)

        assert close(got, (point / math.sqrt(1 + spread ** 2)).expand(N, P), tol=0.02)


def test_cone_average_steadies_a_measure_against_a_noisy_direction():
    """What it is for, and the case where it earns its keep.

    v is an estimate. Draw a fresh noisy one, ask the same question, and the answer moves. Average
    over a cone the size of the uncertainty and it moves less. `banzhaf` is the measure to show it
    on because it is strongly nonlinear in v -- it evaluates the readout at a point that depends on
    v -- and averaging a nonlinear function over a neighbourhood is the only case where averaging
    can do anything. On `directional_derivative`, which is exactly linear in v, the same call is a
    shrinkage by 1 / sqrt(1 + spread^2) and buys nothing at all.
    """
    torch.manual_seed(11)
    width = 32
    head = torch.nn.Sequential(torch.nn.Linear(width, 32), torch.nn.Tanh(),
                               torch.nn.Linear(32, 1))

    def g(z):
        return head(z).squeeze(-1).sum(1)

    x = torch.randn(N, P, width)
    star = torch.randn(width)
    star = star / star.norm()

    tilt = 0.3
    raw, cone = [], []

    for _ in range(20):
        hat = star + tilt * torch.randn_like(star) / width ** 0.5
        hat = hat / hat.norm()

        raw.append(float(D.banzhaf(g, x, hat).detach().mean()))
        cone.append(float(D.cone_average(D.banzhaf, g, x, hat, spread=tilt, samples=48).detach().mean()))

    assert torch.tensor(cone).std() < 0.95 * torch.tensor(raw).std()


def test_subspace_map_runs_a_measure_once_per_direction():
    g, x, _, _ = problem()

    V = torch.eye(C)[:3]
    got = D.subspace_map(D.gradinput, g, x, V)

    assert got.shape == (N, P, 3)
    assert close(got[..., 0], D.gradinput(g, x, V[0]))


@pytest.mark.parametrize('name', EVERY)
def test_every_measure_is_differentiable_back_into_the_readout(name):
    """The penalty use, not the measurement use: the value has to carry a gradient to whatever
    produced the readout, or it cannot be trained against."""
    torch.manual_seed(7)
    head = torch.nn.Linear(C, 1)

    x = torch.randn(N, P, C)
    v = torch.randn(C)
    v = v / v.norm()

    def g(z):
        return head(z).squeeze(-1).sum(1)

    measure(name)(g, x, v).pow(2).mean().backward()

    assert head.weight.grad is not None
    assert head.weight.grad.abs().sum() > 0




def sphere_average(dg, x, width, draws=20000, pair=False):
    """Monte Carlo over u uniform on the unit sphere, which is what the closed forms claim."""
    torch.manual_seed(0)
    total = 0.0

    for _ in range(draws):
        u = torch.randn(width)
        u = u / u.norm()
        along = dg @ u

        total = total + (along * (x @ u)).pow(2) if pair else total + along.pow(2)

    return total / draws


def test_baseline_directional_derivative_is_the_sphere_average_of_the_squared_slope():
    """E over random unit u of <dg, u>^2 = ||dg||^2 / c, since E[u u^T] = I / c. If this constant
    is wrong every share is wrong by a factor, and nothing else in the suite would notice."""
    torch.manual_seed(0)
    width = 8
    dg = torch.randn(1, 1, width)

    got = D.baseline_directional_derivative(dg, width)
    want = sphere_average(dg, None, width)

    assert torch.allclose(got, want, rtol=0.05)


def test_baseline_gradinput_is_the_spheres_fourth_moment():
    """(||dg||^2 ||x||^2 + 2 <dg, x>^2) / (c (c + 2)). The cross term is the part a wrong
    derivation drops, and dropping it is invisible until dg and x are correlated."""
    torch.manual_seed(0)
    width = 8
    dg, x = torch.randn(1, 1, width), torch.randn(1, 1, width)

    got = D.baseline_gradinput(dg, x, width)
    want = sphere_average(dg, x, width, pair=True)

    assert torch.allclose(got, want, rtol=0.05)


def test_baseline_gradinput_keeps_its_cross_term():
    """Aligned dg and x make the cross term two thirds of the answer, so a version that dropped it
    would be off by a factor of three exactly where a real readout lives."""
    width = 8
    dg = torch.ones(1, 1, width)

    aligned = float(D.baseline_gradinput(dg, dg, width))
    without = float(dg.pow(2).sum(-1) * dg.pow(2).sum(-1) / (width * (width + 2)))

    assert aligned == pytest.approx(3 * without, rel=1e-6)


def test_baseline_gradinput_is_homogeneous_in_the_activation():
    """Why a target survives a change of activation scale: numerator and reference move together."""
    torch.manual_seed(2)
    width = 16
    dg, x = torch.randn(3, 1, width), torch.randn(3, 1, width)

    for k in (10.0, 100.0):
        assert torch.allclose(D.baseline_gradinput(dg, k * x, width),
                              k ** 2 * D.baseline_gradinput(dg, x, width), rtol=1e-5)


def span(seed=0, k=3):
    """An orthonormal basis (k, C), for the measures that accept one."""
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(C, k))

    return q.T


def test_project_out_with_a_basis_is_the_fold_over_its_atoms():
    """Handed orthonormal rows, one call removes the whole span: the same tensor as removing the
    atoms one at a time, and the identity on a direction orthogonal to all of them."""
    _, x, _, _ = problem()
    V = span()
    folded = x.clone()

    for v in V:
        folded = D.project_out(folded, v)

    other = torch.linalg.qr(torch.cat([V, torch.randn(1, C)]).T)[0].T[-1]

    assert close(D.project_out(x, V), folded)
    assert float((D.project_out(x, V) @ V.T).abs().max()) < 1e-5
    assert close(D.project_out(x, V) @ other, x @ other)
    assert D.project_out(x[:, 0], V).shape == (N, C)


def test_banzhaf_over_a_span_is_gradinput_summed_over_its_atoms():
    """On a linear readout the erasure of a span is exact to first order: sum_j <w, q_j> <x_p, q_j>,
    summed over positions. Zero when w is orthogonal to the span."""
    g, x, _, w = problem()
    V = span()
    want = ((x @ V.T) * (w @ V.T)).sum((1, 2))

    assert close(D.banzhaf(g, x, V).squeeze(1), want)
    assert close(D.banzhaf(g, x, V[:1]), D.banzhaf(g, x, V[0]))

    w_off = w - (w @ V.T) @ V
    assert float(D.banzhaf(lambda z: (z @ w_off).sum(1), x, V).abs().max()) < 1e-4


def test_banzhaf_over_a_span_lets_atoms_cancel_where_subspace_map_cannot():
    """Two atoms read with opposite sign on an input at +1 and -1 along them: removing the span
    together changes nothing, while reading one atom at a time counts two violations."""
    V = span(k=2)
    w = V[0] + V[1]
    x = (V[0] - V[1]).expand(N, P, C).clone()

    def g(z):
        return (z @ w).sum(1)

    assert float(D.banzhaf(g, x, V).pow(2).sum()) < 1e-8
    assert float(D.subspace_map(D.banzhaf, g, x, V).pow(2).sum()) > 1.0
