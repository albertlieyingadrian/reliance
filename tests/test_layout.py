"""The plumbing: any tap into (n, p, c) and back, and the flat matrix an estimator reads."""
import pytest
import torch

from reliance import layout


N, P, C = 6, 8, 5


@pytest.mark.parametrize('shape', [(N, C), (N, P, C), (N, C, 3, 4)])
def test_positions_and_restore_are_inverses(shape):
    act = torch.randn(*shape)

    assert torch.equal(layout.restore(layout.positions(act), shape), act)


@pytest.mark.parametrize('shape,want', [((N, C), (N, 1, C)), ((N, P, C), (N, P, C)),
                                        ((N, C, 3, 4), (N, 12, C))])
def test_positions_lands_on_the_canonical_shape(shape, want):
    assert layout.positions(torch.randn(*shape)).shape == want


def test_a_conv_map_keeps_its_channels_together():
    """The one that is easy to get wrong: a conv map is channel-first, so a naive reshape
    interleaves positions and channels and every method silently reads noise."""
    act = torch.randn(N, C, 3, 4)
    flat = layout.positions(act)

    assert torch.equal(flat[0, 0], act[0, :, 0, 0])
    assert torch.equal(flat[0, 5], act[0, :, 1, 1])


@pytest.mark.parametrize('shape', [(N, C), (N, P, C), (N, C, 3, 4)])
def test_points_is_one_row_per_observation(shape):
    """Whatever the tap, an estimator gets a matrix of vectors of width C."""
    flat = layout.points(torch.randn(*shape))

    assert flat.shape == (torch.Size(shape).numel() // C, C)


def test_points_keeps_a_pair_aligned_row_for_row():
    """A paired estimator subtracts pos[i] - neg[i] after flattening, so the two must agree."""
    a = torch.randn(N, C, 3, 4)
    b = a.clone()
    b[2, :, 1, 1] += 5.0

    moved = (layout.points(a) - layout.points(b)).norm(dim=1)

    assert int(moved.argmax()) == 2 * 12 + 5


def test_orient_flips_only_the_first_row():
    V = torch.eye(4)[:3]

    flipped = layout.orient(V, -torch.eye(4)[0])

    assert torch.equal(flipped[0], -V[0])
    assert torch.equal(flipped[1:], V[1:])
    assert torch.equal(layout.orient(V, torch.eye(4)[0]), V)


def test_normalize_is_a_unit_line_that_keeps_its_sign():
    v = torch.tensor([-3.0, 4.0])

    got = layout.normalize(v)

    assert got.shape == (1, 2)
    assert got.norm() == pytest.approx(1.0, abs=1e-6)
    assert float(got[0] @ v) > 0


def test_gauge_puts_the_average_norm_at_root_d():
    """What makes one learning rate and one penalty coefficient mean the same thing everywhere:
    after gauging, every coordinate is O(1) whatever the activation arrived at."""
    torch.manual_seed(0)

    for scale in (1e-3, 1.0, 1e3):
        rows = layout.gauge(scale * torch.randn(200, C))

        assert float(rows.norm(dim=-1).mean()) == pytest.approx(C ** 0.5, rel=1e-4)


def test_gauge_moves_no_direction():
    """The whole argument for gauging: it is one scalar, so it cannot rotate anything. If it
    could, every estimator downstream would be reading a different question at a different norm."""
    torch.manual_seed(1)
    x = torch.randn(50, C)

    before = x / x.norm(dim=-1, keepdim=True)
    after = layout.gauge(x)
    after = after / after.norm(dim=-1, keepdim=True)

    assert torch.allclose(before, after, atol=1e-5)
