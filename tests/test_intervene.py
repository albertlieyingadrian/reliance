"""The four rewrites, on a toy where the coordinate along v carries the label and nothing else does.

Each is checked on what it claims and on what separates it from its neighbour: `pps` shifts by a
constant, `noise` destroys the coordinate, `shuffle` keeps its marginal exactly and destroys only
the pairing with the label, `project_out` deletes it. The shared property is that all four move the
activation along v and nowhere else, which is what makes them one family.
"""
import torch

from reliance import intervene as I


N, D = 2000, 32


def toy(n=N, seed=0):
    """x whose coordinate along v carries the label at d' = 4.2, isotropic noise elsewhere."""
    torch.manual_seed(seed)
    v = torch.eye(D)[0]
    y = (torch.rand(n) < 0.5).float()

    return torch.randn(n, 1, D) + 3.0 * (2 * y - 1)[:, None, None] * v, y, v


def auc(coord, y):
    """Share of (positive, negative) pairs the coordinate ranks correctly. 0.5 is chance."""
    pos, neg = coord[y == 1], coord[y == 0]

    return float((pos[:, None] > neg[None, :]).float().mean())


def along(x, v):
    return (x @ v).squeeze(1)


def test_the_toy_is_readable_before_anything_touches_it():
    """Asserted first: an intervention that destroys nothing looks identical to one that works
    on data where there was nothing to destroy."""
    x, y, v = toy()

    assert auc(along(x, v), y) > 0.95


def test_pps_is_exactly_a_constant_shift():
    """The defining property, and the reason it cannot help against a feature being read: every
    example moves by the same eps v, so no difference between examples changes."""
    x, _, v = toy()

    for eps in (0.1, 1.5, 100.0):
        assert torch.allclose(I.pps(x, v, eps) - x, eps * v, atol=1e-5)


def test_noise_makes_the_coordinate_unreadable():
    x, y, v = toy()

    assert auc(along(I.noise(x, v, sigma=20.0), v), y) < 0.65


def test_shuffle_keeps_the_marginal_exactly_and_loses_only_the_pairing():
    """The difference from `noise`, and why both are here. A permutation cannot change the
    multiset of coordinates, so the distribution along v survives intact -- what goes is which
    example carried which value. `noise` destroys the distribution as well."""
    x, y, v = toy()

    was = along(x, v)
    now = along(I.shuffle(x, v), v)

    assert torch.allclose(was.sort().values, now.sort().values, atol=1e-4)
    assert auc(now, y) < 0.65


def test_noise_does_not_keep_the_marginal():
    """The other half of the contrast above, or the previous test proves nothing."""
    x, _, v = toy()

    was = along(x, v)
    now = along(I.noise(x, v, sigma=20.0), v)

    assert not torch.allclose(was.sort().values, now.sort().values, atol=1e-2)


def test_project_out_deletes_the_coordinate():
    x, _, v = toy()

    assert float(along(I.project_out(x, v), v).abs().max()) < 1e-5


def test_every_intervention_moves_the_activation_along_v_and_nowhere_else():
    """What makes these one family: each rewrites one coordinate and leaves the complement alone.

    Checked on a direction orthogonal to v, where every one of them has to be the identity.
    """
    x, _, v = toy()
    other = torch.eye(D)[7]

    for name, rewritten in (('pps', I.pps(x, v)), ('noise', I.noise(x, v, sigma=5.0)),
                            ('shuffle', I.shuffle(x, v)), ('project_out', I.project_out(x, v))):
        assert torch.allclose(along(rewritten, other), along(x, other), atol=1e-5), name


def test_noise_reads_the_same_at_either_rank():
    """One draw per example whatever sits between the example axis and the channel. Hardcoding two
    trailing axes turned an (n, d) input into (n, 1, d) silently, which nothing downstream said."""
    torch.manual_seed(1)
    v = torch.eye(D)[0]

    for shape in ((64, D), (64, 5, D), (64, 3, 4, D)):
        assert I.noise(torch.randn(*shape), v).shape == torch.Size(shape)
