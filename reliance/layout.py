"""Control shape functionality.
"""
import torch
from einops import rearrange

EPS = 1e-12


def positions(act):
    """Any tap as (n, p, c). A tap with no positions is p = 1."""
    if act.ndim == 4:
        return rearrange(act, 'n c h w -> n (h w) c')

    if act.ndim == 2:
        return rearrange(act, 'n c -> n 1 c')

    return act


def points(act):
    """Any tap as (N, c): one row per position of per example, which is one observation.

    An estimator wants a set of points in activation space and does not care which image a point
    came from. Pairing survives this row for row, so a paired estimator can still subtract.
    """
    return rearrange(positions(act), 'n p c -> (n p) c')


def restore(flat, shape):
    """The inverse of `positions`, given the shape it was called on."""
    if len(shape) == 4:
        _, _, h, w = shape
        return rearrange(flat, 'n (h w) c -> n c h w', h=h, w=w)

    if len(shape) == 2:
        return rearrange(flat, 'n 1 c -> n c')

    return flat


def normalize(v):
    """One direction as a unit-norm (1, d) line. The sign is left alone."""
    return (v / v.norm().clamp(min=EPS))[None]


def gauge(x):
    """Rows rescaled so their average norm is sqrt(d), which puts every coordinate at O(1).

    Isotropic, so no direction moves, and it is what lets one learning rate or one penalty
    coefficient mean the same thing at every activation norm. Without it an optimiser tuned where
    ||x|| is 1 is undertrained where it is 100, and that difference reads as a property of the
    estimator rather than of the fit.
    """
    return x * x.shape[-1] ** 0.5 / x.norm(dim=-1).mean().clamp(min=EPS)


def orient(V, towards):
    """Flip row 0 of a basis to agree with `towards`.

    A span has no sign, but the first direction of one usually does: `intervene.pps` pushes along
    it and pushing the other way is a different and, on its authors' account, harmful method. QR
    and SVD both return an arbitrary sign, so it is fixed here rather than hoped for.
    """
    if float(V[0] @ towards) < 0:
        return torch.cat([-V[:1], V[1:]])

    return V
