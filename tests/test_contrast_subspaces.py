"""What each estimator finds, on data built so the answer is known in closed form.

Every case is made where it is used, because each one is about a different shape: a displacement
smeared over several directions, a displacement plus noise orthogonal to it, a loud one shadowed
by a quiet satellite, a confounded one. The first test is the control -- with no structure in the
way every estimator works, so the rest measures structure and not a bug.
"""
import pytest
import torch

from reliance import layout as L
from reliance import contrast_subspaces as S


D = 64
N = 400


def fitted(estimate):
    """Anything that trains a classifier costs real time here, so it is marked and skippable."""
    return pytest.param(estimate, marks=pytest.mark.slow)


SINGLE_DIRECTION = [S.dom, S.fisher, fitted(S.linear_probe), fitted(S.svm)]
DISCRIMINANTS = [S.fisher, fitted(S.linear_probe), fitted(S.svm)]


def plain(n=800, seed=0):
    """One direction, isotropic noise, nothing in the way. Paired, so anything may be run on it."""
    torch.manual_seed(seed)
    neg = torch.randn(n, D)

    return neg + 2.0 * torch.eye(D)[0], neg, torch.eye(D)[0]


def cone(n=N, width=4, spread=1.0, seed=1):
    """A displacement spread over `width` directions, each firing at mean one and varying.

    Every direction shifts the mean, so a covariance method can find them all; the mean of the
    whole is one line through the middle of the cone, which is all a difference of means returns.
    """
    torch.manual_seed(seed)
    U = torch.eye(D)[:width]
    neg = torch.randn(n, D)
    weight = 1.0 + spread * torch.randn(n, width)

    return neg + weight @ U, neg, L.normalize(U.sum(0))[0], U


def offset(n=N, along=3.0, around=1.0, extra=4, seed=2):
    """A fixed displacement plus noise strictly orthogonal to it. What centring destroys."""
    torch.manual_seed(seed)
    neg = torch.randn(n, D)
    noise = around * torch.randn(n, extra) @ torch.eye(D)[1:1 + extra]

    return neg + along * torch.eye(D)[0] + noise, neg, torch.eye(D)[0]


def loud_and_quiet(n=N, loud=8.0, tell=0.5, seed=3):
    """A loud, badly separated displacement, and a quiet tight satellite that moves with it.

    The core carries sixteen times the content at a spread as wide as its mean, so mu ranks it
    first and mu / sigma^2 ranks the satellite first by a factor of some hundreds.
    """
    torch.manual_seed(seed)
    core, mark = torch.eye(D)[0], torch.eye(D)[1]
    pos = (0.1 * torch.randn(n, D) + (loud + loud * torch.randn(n, 1)) * core
           + (tell + 0.02 * torch.randn(n, 1)) * mark)

    return pos, 0.1 * torch.randn(n, D), L.normalize(loud * core + tell * mark)[0], mark


def confounded(n=2000, rho=0.8, seed=4):
    """A concept whose partner fires with it a share (1 + rho) / 2 of the time."""
    torch.manual_seed(seed)
    on = torch.rand(n) < 0.5
    partner = on == (torch.rand(n) < (1 + rho) / 2)
    x = (0.3 * torch.randn(n, D) + on[:, None] * torch.eye(D)[0]
         + partner[:, None] * torch.eye(D)[1])

    return x, on.float(), partner.float(), torch.eye(D)[0]


def captured(V, pos, neg):
    """Share of the actual displacement lying in V, per row. What a projection would remove.

    An expectation of a ratio, not a ratio of expectations: a line through a cone has a projector
    norm of 1 to the mean and still misses much of any particular move.
    """
    move = pos - neg

    return float(((move @ V.T).norm(dim=1) / move.norm(dim=1).clamp(min=1e-12)).mean())


def test_projector_norm_is_one_inside_the_subspace_and_zero_outside():
    V = torch.eye(8)[:3]

    assert S.projector_norm(V, torch.eye(8)[0]) == pytest.approx(1.0, abs=1e-5)
    assert S.projector_norm(V, torch.eye(8)[5]) == pytest.approx(0.0, abs=1e-5)
    half = torch.eye(8)[0] + torch.eye(8)[5]

    assert S.projector_norm(V, half) == pytest.approx(2 ** -0.5, abs=1e-5)


def test_angle_ignores_the_sign():
    torch.manual_seed(0)
    v = torch.randn(8)

    assert S.angle(v, -v) == pytest.approx(0.0, abs=1e-3)
    assert S.angle(v, v) == pytest.approx(0.0, abs=1e-3)


def test_a_direction_keeps_its_sign_and_points_from_neg_to_pos():
    """`intervene.pps` pushes along v and the sign *is* the method, so an estimate is oriented."""
    pos, neg, _ = plain()
    gap = pos.mean(0) - neg.mean(0)

    assert float(S.dom(pos, neg)[0] @ gap) > 0
    assert float(S.paired_svd(pos, neg)[0] @ gap) > 0


@pytest.mark.parametrize('estimate', [S.dom, S.fisher, S.paired_svd, S.pls,
                                      fitted(S.linear_probe), fitted(S.svm), fitted(S.inlp)])
def test_every_estimator_returns_an_orthonormal_basis(estimate):
    pos, neg, _ = plain(200)
    k = 3 if estimate in (S.paired_svd, S.pls, S.inlp) else 1

    V = estimate(pos, neg, k=k)

    assert V.shape == (k, D)
    assert torch.allclose(V @ V.T, torch.eye(k), atol=1e-4)


@pytest.mark.parametrize('estimate', SINGLE_DIRECTION)
def test_a_single_direction_estimator_refuses_to_pretend_it_is_a_subspace(estimate):
    pos, neg, _ = plain(200)

    with pytest.raises(ValueError):
        estimate(pos, neg, k=4)


@pytest.mark.parametrize('estimate', SINGLE_DIRECTION)
def test_every_estimator_finds_an_isotropic_concept(estimate):
    """The control. One direction, no satellite, no confound, and they all land on it. They come
    apart only once the concept has structure, which is what the rest of the file adds."""
    pos, neg, truth = plain()

    assert S.projector_norm(estimate(pos, neg), truth) > 0.9


def test_dom_is_exactly_the_gap_between_the_two_means():
    pos, neg, _ = plain()

    assert torch.allclose(S.dom(pos, neg), L.normalize(pos.mean(0) - neg.mean(0)), atol=1e-6)


def test_pairing_buys_nothing_at_one_direction():
    """The mean is linear, so the mean of the paired differences *is* the difference of means.

    The premise of the next two: pairing cannot help at k = 1, which is why `paired_svd` is a
    different estimator and why centring the differences is fatal rather than merely wasteful.
    """
    pos, neg, _ = plain()

    paired = L.normalize((pos - neg).mean(0))

    assert S.angle(S.dom(pos, neg), paired) == pytest.approx(0.0, abs=1e-3)


def test_a_single_direction_cannot_hold_a_smeared_concept():
    """A cone of four directions firing together. `dom` returns their average, a good line through
    the middle that still misses a quarter of any particular move."""
    pos, neg, _, _ = cone()

    line = captured(S.dom(pos, neg), pos, neg)
    whole = captured(S.paired_svd(pos, neg, k=4), pos, neg)

    assert whole > 0.99
    assert whole > line + 0.05


def test_centring_the_paired_differences_throws_the_answer_away():
    """The mistake experiment 17 measured at 0.925 -> 0.550, as a property rather than a score.

    The noise here is orthogonal to the displacement, so centring removes the displacement exactly
    and leaves a subspace that has nothing to do with it.
    """
    pos, neg, truth = offset()

    kept = S.projector_norm(S.paired_svd(pos, neg, k=1), truth)
    lost = S.projector_norm(S.paired_svd(pos, neg, k=1, center=True), truth)

    assert kept > 0.95
    assert lost < 0.3


def test_the_discriminant_deserts_the_concept_for_its_satellite():
    """Experiment 17's last-placed estimator (0.541 against 0.950), and the mechanism for it.

    Fisher divides the displacement by the covariance. The core varies as much as it means, so the
    division all but deletes it and the satellite is what is left.
    """
    pos, neg, truth, mark = loud_and_quiet()

    found = S.fisher(pos, neg)

    assert S.projector_norm(found, truth) < 0.3
    assert S.projector_norm(found, mark) > 0.8
    assert S.projector_norm(found, mark) > S.projector_norm(found, truth)


@pytest.mark.parametrize('estimate', DISCRIMINANTS)
def test_every_discriminant_loses_ground_to_a_difference_of_means(estimate):
    """Three rows and not one claim, because they do not fail alike: fisher deserts outright,
    while a probe and an SVM only lean, their losses being unregularised in scale."""
    pos, neg, truth, _ = loud_and_quiet()

    assert S.projector_norm(S.dom(pos, neg), truth) > \
           S.projector_norm(estimate(pos, neg), truth) + 0.15


@pytest.mark.parametrize('estimate', DISCRIMINANTS)
def test_the_discriminants_win_the_contest_they_were_built_for(estimate):
    """What makes it a trade rather than a bug: the estimators that repair worst genuinely are
    the better detectors. Same ordering, reversed."""
    pos, neg, _, _ = loud_and_quiet()

    assert S.direction_auc(estimate(pos, neg), pos, neg) > \
           S.direction_auc(S.dom(pos, neg), pos, neg)


def test_removing_what_a_detector_found_removes_almost_no_content():
    """Delete the discriminant and the content is still on the displacement, where training finds
    another reader for it by the next epoch."""
    pos, neg, _, _ = loud_and_quiet()

    read = captured(S.fisher(pos, neg), pos, neg)
    write = captured(S.dom(pos, neg), pos, neg)

    assert read < 0.3
    assert write > 0.7
    assert write > 3 * read


def test_pls_component_one_is_the_difference_of_means():
    """w_1 maximises Cov(Xw, y), so it is proportional to X^T y, which for two balanced groups is
    exactly mean(pos) - mean(neg). PLS generalises `dom`; it does not rival it."""
    pos, neg, _, _ = cone()

    assert S.angle(S.pls(pos, neg, k=1), S.dom(pos, neg)) < 2.0


def test_pls_stays_on_the_side_of_the_displacement():
    """Covariance, not correlation: nothing is divided by variance, so the satellite that captures
    `fisher` does not capture this."""
    pos, neg, truth, _ = loud_and_quiet()

    assert S.projector_norm(S.pls(pos, neg, k=1), truth) > 0.95
    assert S.projector_norm(S.fisher(pos, neg), truth) < 0.3


def test_pls_adds_little_past_its_first_component_on_a_cone():
    """Measured, not assumed, and the reason its docstring hedges. Covariance with y is spent once
    the mean shift is deflated, so a cone that is spread rather than shift stays invisible."""
    pos, neg, _, _ = cone()
    shuffled = neg[torch.randperm(len(neg))]

    assert captured(S.pls(pos, shuffled, k=4), pos, neg) < 0.8
    assert captured(S.paired_svd(pos, neg, k=4), pos, neg) > 0.99


def test_a_difference_of_means_takes_the_confound_with_it():
    x, on, partner, truth = confounded()

    assert S.projector_norm(S.dom(x[on.bool()], x[~on.bool()]), truth) < 0.9


def test_stratifying_on_the_confound_puts_the_direction_back():
    x, on, partner, truth = confounded()
    strata = [(x[on.bool() & (partner == level)], x[~on.bool() & (partner == level)])
              for level in (0.0, 1.0)]

    assert S.projector_norm(S.stratified_dom(strata), truth) > 0.95


def test_adjusted_dom_is_exact_when_the_confound_is_linear():
    """A regression seeing both indicators returns the concept's own coefficient however strongly
    they correlate, up to collinearity."""
    x, on, partner, truth = confounded()

    assert S.projector_norm(S.adjusted_dom(x, on, partner), truth) > 0.95


@pytest.mark.slow
def test_inlp_chases_the_loud_direction_and_leaves_the_readable_one():
    """A limitation written down rather than tuned away, because it is diagnosable.

    `inlp` should trade content away and win on unreadability; it does neither. Its rounds are
    probe fits on raw activations, so they chase magnitude and never reach the tight satellite.
    Standardise before each fit and this should break -- it is here to fail, not to be satisfied.
    """
    pos, neg, _, _ = loud_and_quiet()

    unreadable = S.inlp(pos, neg, k=4)
    content = S.paired_svd(pos, neg, k=4)

    left = lambda V: S.direction_auc(S.linear_probe(*residual(V, pos, neg)), *residual(V, pos, neg))

    assert captured(content, pos, neg) > captured(unreadable, pos, neg)
    assert left(unreadable) > left(content)


def residual(V, *groups):
    """Each group with the span of V projected out. What is left for a probe to read."""
    return tuple(g - (g @ V.T) @ V for g in groups)


def overlapping_atoms(n=800, tilt=0.4, seed=8):
    """Six unit atoms, each an axis tilted toward one shared direction, two used per difference.

    Mutual cosine is tilt^2 / (1 + tilt^2), so 0.14 here -- inside the coherence sparse coding
    needs. Measured, recovery is 0.997 at 0.14, 0.81 at 0.27 and 0.685 at 0.50, where an SVD of
    the same data does better. The bound is about 1 / (2 * active - 1), which is 1/3 at active 2.
    """
    torch.manual_seed(seed)
    A = torch.stack([torch.eye(D)[i + 1] + tilt * torch.eye(D)[0] for i in range(6)])
    A = A / A.norm(dim=1, keepdim=True)

    pick = torch.stack([torch.randperm(len(A))[:2] for _ in range(n)])
    weight = 0.3 + 2.7 * torch.rand(n, 2)
    move = weight[:, 0, None] * A[pick[:, 0]] + weight[:, 1, None] * A[pick[:, 1]]

    return torch.randn(n, D) * 0.05 + move, torch.randn(n, D) * 0.05, A


def best_match(found, truth):
    """For each true atom, the closest thing among `found`, by absolute cosine."""
    found = found / found.norm(dim=1, keepdim=True)
    truth = truth / truth.norm(dim=1, keepdim=True)

    return (truth @ found.T).abs().max(dim=1).values


def test_cca_on_a_binary_label_is_fisher():
    """Written down so nobody adds it twice. Sigma_total = Sigma_within + c mu mu^T, so by
    Sherman-Morrison the two solves give the same direction: LDA, CCA and Fisher coincide here."""
    pos, neg, _ = plain()

    x = torch.cat([pos, neg]).double()
    y = torch.cat([torch.ones(len(pos)), -torch.ones(len(neg))]).double()
    x, y = x - x.mean(0), y - y.mean()
    total = torch.linalg.solve(x.T @ x / len(x), x.T @ y).float()

    within = torch.cat([pos - pos.mean(0), neg - neg.mean(0)]).double()
    mu = (pos.mean(0) - neg.mean(0)).double()
    inner = torch.linalg.solve(within.T @ within / len(within), mu).float()

    assert S.angle(total, inner) < 0.01


def test_sparse_delta_returns_the_atoms_an_orthogonal_basis_can_only_span():
    """An SVD of the same differences spans these six and cannot be them: its rows are a rotation
    inside the span, with no reason to land on the directions that generated it."""
    pos, neg, truth = overlapping_atoms()

    atoms, _ = S.sparse_delta(pos, neg)
    rotated = S.paired_svd(pos, neg, k=6)

    assert float(best_match(atoms, truth).min()) > 0.95
    assert float(best_match(rotated, truth).min()) < 0.7


def test_top_k_prunes_the_spare_slots_where_l1_could_not():
    """The reason this codes by top-k rather than by an L1 penalty.

    L1 charges for the size of a loading, not the number of atoms carrying it: split z between two
    copies of a direction and |z/2| + |z/2| is still |z|, so duplicates never merge -- measured, it
    left 42% of the energy in the spare slots at every coefficient tried. A top-k slot is a real
    budget, so twenty slots collapse onto the six atoms that exist.
    """
    pos, neg, truth = overlapping_atoms()

    _, loads = S.sparse_delta(pos, neg)
    energy = loads.pow(2).sum(0)

    assert bool((energy[:-1] >= energy[1:]).all())
    assert float(energy[len(truth):].sum() / energy.sum()) < 0.05
    assert int((energy > 0.01 * energy.max()).sum()) == len(truth)     # ten slots, six survive

def test_the_decoder_writes_and_the_encoder_reads():
    """Experiment 12's alignment of 0.017, in closed form on a two-atom dictionary.

    Two atoms 26 degrees apart. The encoder is the decoder's pseudo-inverse, so the row recovering
    atom 0 is the one orthogonal to atom *1*: shaped by what it must ignore, not what it is for.
    """
    a0 = torch.tensor([1.0, 0.0, 0.0])
    a1 = torch.tensor([0.9, (1 - 0.81) ** 0.5, 0.0])
    decoder = torch.stack([a0, a1])
    encoder = torch.linalg.pinv(decoder).T

    z = torch.tensor([[2.0, 3.0], [1.0, -1.0]])

    assert torch.allclose((z @ decoder) @ encoder[0], z[:, 0], atol=1e-5)
    assert S.projector_norm(S.orthobasis(decoder[[0]]), a0) == pytest.approx(1.0, abs=1e-5)
    assert S.projector_norm(S.orthobasis(encoder[[0]]), a0) < 0.6


def dom_svd_deltas(n=400, seed=0, spread=3.0, noise=0.1):
    """Differences that shift by one along e0 and vary by `spread` along e1, plus a little
    isotropic noise: the mean is on e0, the second moment is largest on e1."""
    torch.manual_seed(seed)
    neg = torch.randn(n, D)
    pos = (neg + torch.eye(D)[0] + spread * torch.randn(n, 1) * torch.eye(D)[1]
           + noise * torch.randn(n, D))

    return pos, neg


def test_dom_svd_at_rank_one_is_dom():
    pos, neg = dom_svd_deltas()

    assert torch.allclose(S.dom_svd(pos, neg, 1), S.dom(pos, neg), atol=1e-5)


def test_dom_svd_holds_the_mean_where_paired_svd_spreads_it():
    """The differences vary three times more along e1 than they shift along e0, so `paired_svd`
    puts its first atom on the variance; `dom_svd` keeps the mean as an exact atom."""
    pos, neg = dom_svd_deltas()
    mean = pos.mean(0) - neg.mean(0)

    assert S.angle(S.dom_svd(pos, neg, 2)[0], mean) < 1e-3
    assert S.angle(S.paired_svd(pos, neg, 2)[0], torch.eye(D)[0]) > 45.0
    assert S.projector_norm(S.dom_svd(pos, neg, 2), mean) > 0.9999


def test_dom_svd_rows_are_orthonormal_and_the_rest_finds_the_spread():
    pos, neg = dom_svd_deltas()
    V = S.dom_svd(pos, neg, 3)

    assert torch.allclose(V @ V.T, torch.eye(3), atol=1e-5)
    assert S.angle(V[1], torch.eye(D)[1]) < 10.0


def test_dom_svd_refuses_to_pad_from_the_nullspace():
    """Differences of exact rank two have one direction past their mean; a third atom would be
    an arbitrary vector, so asking for it is an error and not a silent completion."""
    pos, neg = dom_svd_deltas(noise=0.0)

    with pytest.raises(ValueError):
        S.dom_svd(pos, neg, 3)

    assert S.dom_svd(pos, neg, 2).shape == (2, D)
