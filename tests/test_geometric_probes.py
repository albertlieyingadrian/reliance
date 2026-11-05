"""One setting per probe, built so the shape it fits is the shape the data has.

Each case is one a hyperplane cannot answer: a ball whose classes share a mean, two clusters a
single ball cannot cover without covering the gap, a plane the class spreads over symmetrically,
a slab whose outside is a union of two half-spaces. `linear_probe` is scored alongside every one
of them, at chance in all four, which is what makes the comparison worth printing.
"""
import pytest
import torch
import torch.nn.functional as F

from reliance import geometric_probes as G
from reliance import contrast_subspaces as S


D = 64
N = 400


def auc(scores, n_pos):
    """Share of (positive, negative) pairs the score ranks correctly. 0.5 is chance."""
    pos, neg = scores[:n_pos], scores[n_pos:]

    return float((pos[:, None] > neg[None, :]).float().mean())


def ball_score(points, width, x):
    """Noisy-OR membership of the union of balls, the object `region_probe` returns."""
    distance = (x[:, None, :] - points[None]).pow(2).sum(-1) / width

    return 1.0 - (1.0 - torch.exp(-distance)).prod(-1)


def draw(seed, positives, negatives):
    """Stack a case and hand back (pos, neg, all, n_pos)."""
    torch.manual_seed(seed)
    pos, neg = positives(), negatives()

    return pos, neg, torch.cat([pos, neg]), len(pos)


def test_spread_points_picks_real_rows_and_covers_the_data():
    """The init every region probe starts from: real examples, and not all in one corner."""
    torch.manual_seed(0)
    x = torch.cat([torch.randn(50, D) + 5, torch.randn(50, D) - 5])

    picked = G.spread_points(x, 4)
    matches = (picked[:, None, :] == x[None]).all(-1).any(-1)

    assert picked.shape == (4, D)
    assert bool(matches.all())
    assert float(torch.cdist(picked, picked).max()) > 8.0


def test_nothing_in_the_file_draws_a_random_number():
    """The seedless guarantee, asserted rather than trusted: same data, same answer, twice."""
    pos, neg, _, _ = draw(1, lambda: torch.randn(200, D) * 0.3, lambda: torch.randn(200, D))

    assert torch.equal(G.region_probe(pos, neg, 3)[0], G.region_probe(pos, neg, 3)[0])
    assert torch.equal(G.subspace_probe(pos, neg, 2), G.subspace_probe(pos, neg, 2))
    assert torch.equal(G.manifold_probe(pos, neg, 3)[0], G.manifold_probe(pos, neg, 3)[0])
    assert torch.equal(G.linear_probe(pos, neg), G.linear_probe(pos, neg))


def test_point_probe_finds_a_ball_the_two_classes_share_a_mean_of():
    """A tight cloud inside a wide one. Both means are the origin, so a direction has nothing to
    point at and the only thing that separates them is a distance."""
    pos, neg, x, n_pos = draw(2, lambda: torch.randn(N, D) * 0.3, lambda: torch.randn(N, D))

    point, width = G.point_probe(pos, neg)

    assert float(point.norm()) < 1.5
    assert auc(ball_score(point[None], width[None], x), n_pos) > 0.95
    assert auc(x @ G.linear_probe(pos, neg)[0], n_pos) < 0.65


def test_region_probe_covers_two_clusters_one_ball_cannot():
    """Positives sit at both ends of an axis with negatives between them, so any single ball wide
    enough to hold both holds the negatives too."""
    far = 6.0 * torch.eye(D)[0]
    pos, neg, x, n_pos = draw(3,
                              lambda: torch.cat([torch.randn(N // 2, D) * 0.3 + far,
                                                 torch.randn(N // 2, D) * 0.3 - far]),
                              lambda: torch.randn(N, D) * 0.3)

    points, width = G.region_probe(pos, neg, 2)
    one, single = G.point_probe(pos, neg)

    assert float(torch.cdist(points, torch.stack([far, -far])).min(0).values.max()) < 2.0
    assert auc(ball_score(points, width, x), n_pos) > 0.95
    assert auc(ball_score(one[None], single[None], x), n_pos) < 0.8


def test_subspace_probe_recovers_the_plane_the_class_spreads_over():
    """Positives ride a ring inside one plane and negatives do not, with both classes noisy
    everywhere else. The ring is symmetric, so no single direction of it means anything."""
    def positives():
        angle = torch.rand(N) * 6.2832
        ring = torch.zeros(N, D)
        ring[:, 0], ring[:, 1] = 4 * angle.cos(), 4 * angle.sin()

        return ring + torch.randn(N, D) * 0.3

    pos, neg, x, n_pos = draw(4, positives, lambda: torch.randn(N, D) * 0.3)

    V = G.subspace_probe(pos, neg, 2)
    plane = torch.eye(D)[:2]

    found = auc((x @ V.T).norm(dim=-1), n_pos)
    line = auc(x @ G.linear_probe(pos, neg)[0], n_pos)

    assert S.projector_norm(V, plane[0]) > 0.9
    assert S.projector_norm(V, plane[1]) > 0.9
    assert found > 0.95
    assert found > line + 0.25


def test_manifold_probe_carves_the_union_of_two_half_spaces():
    """Positives are outside a slab and negatives inside it, so the region wanted is a union of
    two opposed half-spaces -- the smallest thing a single hyperplane provably cannot be."""
    def outside():
        x = torch.randn(N, D) * 0.3
        x[:, 0] = torch.sign(torch.randn(N)) * (2.0 + torch.rand(N))

        return x

    def inside():
        x = torch.randn(N, D) * 0.3
        x[:, 0] = torch.rand(N) - 0.5

        return x

    pos, neg, x, n_pos = draw(5, outside, inside)

    weight, bias = G.manifold_probe(pos, neg, 2)
    along = (weight / weight.norm(dim=1, keepdim=True)) @ torch.eye(D)[0]

    found = auc(F.gelu(x @ weight.T + bias).sum(-1), n_pos)
    line = auc(x @ G.linear_probe(pos, neg)[0], n_pos)

    assert found > 0.9
    assert found > line + 0.25
    assert float(along.abs().min()) > 0.6      # both units point along the slab's own axis
    assert float(along[0] * along[1]) < 0      # and they point opposite ways


def test_a_subspace_probe_can_say_outside():
    """The trainable bias before the sigmoid. Without it the score is a norm, never negative, so
    every negative carries an irreducible log 2 and the probe can only say inside or unsure."""
    pos, neg, x, _ = draw(6, lambda: torch.randn(N, D) + 3, lambda: torch.randn(N, D))

    model = G.Subspace(G.leading_directions(pos, neg, 2))
    stack, target = G.stacked(pos, neg)
    G.fit_params(model, stack, target, F.binary_cross_entropy_with_logits)

    with torch.no_grad():
        scores = model(stack)

    assert float(scores.min()) < 0.0
    assert float(torch.sigmoid(scores.min())) < 0.4


def test_the_difference_of_means_start_keeps_inlp_orthogonal():
    """`linear_probe` starts from the difference of means, which lies in whatever span is left
    after `inlp` projects a round out -- so the rounds stay orthogonal without being forced."""
    pos, neg, _, _ = draw(7, lambda: torch.randn(200, D) + 2, lambda: torch.randn(200, D))

    V = S.inlp(pos, neg, k=4)

    assert V.shape == (4, D)
    assert torch.allclose(V @ V.T, torch.eye(4), atol=1e-4)


def test_region_probe_fits_an_ellipsoid_when_asked():
    """The anisotropic branch, on the only data that can tell it from the isotropic one.

    Positives fill a slab: wide along one axis, tight along another. A ball wide enough to hold
    the length swallows negatives across the width, so the isotropic fit has to trade one against
    the other; a learned metric per point does not.
    """
    def slab():
        z = torch.randn(N, D) * 0.15
        z[:, 0] = 4.0 * (torch.rand(N) - 0.5)

        return z

    def around():
        z = torch.randn(N, D) * 0.15
        z[:, 0] = 4.0 * (torch.rand(N) - 0.5)
        z[:, 1] = torch.sign(torch.randn(N)) * (0.8 + 0.4 * torch.rand(N))

        return z

    pos, neg, x, n_pos = draw(9, slab, around)

    points, metric = G.region_probe(pos, neg, 2, anisotropic=True, rank=4)
    round_points, width = G.region_probe(pos, neg, 2)

    stretched = G.Prototypes(points, anisotropic=True, rank=4)
    with torch.no_grad():
        stretched.L.copy_(metric)
        stretched.points.copy_(points)
        found = auc(stretched(x), n_pos)

    assert metric.shape == (2, 4, D)
    assert found > auc(ball_score(round_points, width, x), n_pos)


def test_leading_directions_is_bounded_by_the_rows_as_well_as_the_width():
    """A thin SVD returns min(n, d) right singular vectors, so a guard on the width alone lets an
    over-large k through and hands back fewer rows than asked. `Subspace` takes its rank from that
    length, so the silent version is a subspace of the wrong size rather than an error."""
    rows, wide = 50, D
    pos, neg, _, _ = draw(10, lambda: torch.randn(rows, wide), lambda: torch.randn(rows, wide))

    assert G.leading_directions(pos, neg, rows).shape == (rows, wide)

    with pytest.raises(ValueError):
        G.leading_directions(pos, neg, rows + 1)        # more than the rows allow

    with pytest.raises(ValueError):
        G.leading_directions(pos, neg, wide + 1)        # more than the width allows
