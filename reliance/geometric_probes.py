"""Library of geometric probes.

Logistic regression and the hinge ask which direction splits a group in two; point, subspace,
manifold and region probes ask instead which point, span, sum of activated directions, or union of
points it clusters into, and return that object -- a point or a basis where a hyperplane probe
returns a direction. Nothing here draws a random number: every fit starts from the data itself.
"""
import torch
import torch.nn.functional as F

from reliance.layout import EPS, gauge, normalize

STEPS = 300
LR = 3e-3
FLOOR = 1e-6          # a probability clamp, not a division guard
WIDEST = 6.0


def spread_points(x, n):
    """n rows of x that cover it: the most central, then repeatedly the one farthest from those.

    Greedy k-centre, and the reason there is no seed in this file. Random starts make a probe
    answer differently on the same data, and a region is a claim about the data.
    """
    chosen = [int((x - x.mean(0)).norm(dim=1).argmin())]
    gap = (x - x[chosen[0]]).norm(dim=1)

    while len(chosen) < n:
        chosen.append(int(gap.argmax()))
        gap = torch.minimum(gap, (x - x[chosen[-1]]).norm(dim=1))

    return x[chosen]


def leading_directions(pos, neg, k):
    """The k directions along which the positive class spreads, measured from the negative mean.

    Bounded by the rows as well as the width: a thin SVD returns min(n, d) right singular vectors,
    so asking for more silently hands back fewer, and `Subspace` reads its rank off that length.
    """
    if k > min(len(pos), pos.shape[1]):
        raise ValueError(f'asked for {k} directions from {len(pos)} rows '
                         f'in {pos.shape[1]} dimensions')

    _, _, vh = torch.linalg.svd(pos - neg.mean(0), full_matrices=False)

    return vh[:k]


class Prototypes(torch.nn.Module):
    """Points with an isotropic width each, or a learned (rank, d) metric each if anisotropic.

    Combined by noisy-OR, so a sample need only be near one of them. Membership *and* the combined
    score are floored: with four saturated points the product underflows, the score is exactly 1,
    and a negative sitting there costs a loss of 100 at a gradient of 1e12. Log widths are clamped
    so a badly placed point cannot swallow the space to hide itself.
    """

    def __init__(self, points, anisotropic=False, rank=None):
        super().__init__()
        n, d = points.shape
        self.anisotropic = anisotropic
        self.points = torch.nn.Parameter(points.clone())

        if anisotropic:
            self.L = torch.nn.Parameter(torch.eye(rank or d, d, device=points.device)
                                        .unsqueeze(0).repeat(n, 1, 1))
        else:
            self.log_width = torch.nn.Parameter(torch.zeros(n, device=points.device))

    def forward(self, x):
        diff = x[:, None, :] - self.points[None, :, :]          # (batch, n, d)

        if self.anisotropic:
            distance = torch.einsum('bnd,nrd->bnr', diff, self.L).pow(2).sum(-1)
        else:
            distance = diff.pow(2).sum(-1) / self.log_width.clamp(-WIDEST, WIDEST).exp()

        membership = torch.exp(-distance).clamp(FLOOR, 1 - FLOOR)

        return (1.0 - (1.0 - membership).prod(-1)).clamp(FLOOR, 1 - FLOOR)

    def metric(self):
        """The learned width per point, or the full (n, rank, d) map when anisotropic."""
        if self.anisotropic:
            return self.L.detach()

        return self.log_width.clamp(-WIDEST, WIDEST).exp().detach()


class Subspace(torch.nn.Module):
    """k directions, scored by how far x's projection onto their span reaches past a threshold.

    The threshold is a trainable bias before the sigmoid and it is not optional. A norm is never
    negative, so without it the score cannot go below one half and every negative carries an
    irreducible log 2: the model can say "inside" or "unsure", never "outside".
    """

    def __init__(self, directions):
        super().__init__()
        self.basis = torch.nn.Linear(directions.shape[1], len(directions), bias=False)
        self.threshold = torch.nn.Parameter(torch.zeros((), device=directions.device))

        with torch.no_grad():
            self.basis.weight.copy_(directions)

    def forward(self, x):
        return self.basis(x).norm(dim=-1) - self.threshold

    def orthonormal(self):
        q, _ = torch.linalg.qr(self.basis.weight.detach().T)

        return q.T


class Manifold(torch.nn.Module):
    """n activated directions summed into one logit: the shallowest network that carves a region.

    The threshold is there for the reason `Subspace` needs one -- a sum of GELUs is bounded below
    near zero, so without it the score cannot express a confident negative.
    """

    def __init__(self, weight, bias, activation=F.gelu):
        super().__init__()
        self.units = torch.nn.Linear(weight.shape[1], len(weight))
        self.threshold = torch.nn.Parameter(torch.zeros((), device=weight.device))
        self.activation = activation

        with torch.no_grad():
            self.units.weight.copy_(weight)
            self.units.bias.copy_(bias)

    def forward(self, x):
        return self.activation(self.units(x)).sum(-1) - self.threshold


def fit_params(model, x, target, loss, steps=STEPS, lr=LR, weight_decay=0.0):
    """Fit any module's parameters by gradient descent. Full batch, so nothing here is random."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(steps):
        objective = loss(model(x), target)
        optimiser.zero_grad()
        objective.backward()
        optimiser.step()

    return model


def fit_linear(x, target, loss, start, **opts):
    """Fit a linear score on x from the direction `start`, and return its weight, (d,).

    `start` is the difference of means, which `inlp` depends on: after a round projects a direction
    out, neither group has a component along it, so their difference has none and no gradient does.
    x is gauged to mean norm sqrt(d) first, without which `lr` and `steps` mean something different
    at every activation norm -- undertrained at one scale and diverging at another.
    """
    x = gauge(x)

    head = torch.nn.Linear(x.shape[1], 1).to(x.device)

    with torch.no_grad():
        head.weight.copy_(normalize(start))
        head.bias.zero_()

    def scored(scores, wanted):
        return loss(scores.squeeze(-1), wanted)

    return fit_params(head, x, target, scored, **opts).weight.detach()[0]


def stacked(pos, neg, positive=1.0, negative=0.0):
    """The two groups as one matrix and one label vector."""
    x = torch.cat([pos, neg])
    target = torch.cat([torch.full((len(pos),), positive),
                        torch.full((len(neg),), negative)]).to(x.device)

    return x, target


def linear_probe(pos, neg, k=1, **opts):
    """Logistic regression weight as a (1, d) line. Reads the habitat at 0.989, repairs at 0.661:
    rewarded for the axis of least overlap, which is not the axis carrying the content.
    """
    if k != 1:
        raise ValueError('linear_probe returns one direction; use inlp for k > 1')

    x, target = stacked(pos, neg)
    start = pos.mean(0) - neg.mean(0)

    return normalize(fit_linear(x, target, F.binary_cross_entropy_with_logits, start, **opts))


def svm(pos, neg, k=1, reg=1e-3, **opts):
    """Max-margin hyperplane normal as a (1, d) line. TCAV's CAV, under its name."""
    if k != 1:
        raise ValueError('svm returns one direction; use contrast_subspaces.inlp for k > 1')

    x, target = stacked(pos, neg, positive=1.0, negative=-1.0)
    start = pos.mean(0) - neg.mean(0)

    def hinge(score, sign):
        return torch.relu(1.0 - sign * score).mean()

    return normalize(fit_linear(x, target, hinge, start, weight_decay=reg, **opts))


def region_probe(pos, neg, n, anisotropic=False, rank=None, **opts):
    """n points (n, d) and their metric: a region built as the union of n balls or ellipsoids.

    Started from real positives, spread out by `spread_points`, rather than from noise near the
    origin -- a point that begins where the data is has something to move away from.
    """
    x, target = stacked(pos, neg)
    model = Prototypes(spread_points(pos, n), anisotropic, rank).to(x.device)
    fit_params(model, x, target, F.binary_cross_entropy, **opts)

    return model.points.detach(), model.metric()


def point_probe(pos, neg, **opts):
    """A single point and its width. The n = 1 case of `region_probe`, started at the most central
    positive rather than at their mean, so it begins on a real example."""
    points, metric = region_probe(pos, neg, 1, **opts)

    return points[0], metric[0]


def subspace_probe(pos, neg, k, **opts):
    """k directions (k, d), reached directly rather than by k rounds of `inlp`.

    Started from the k directions the positive class actually spreads along, and orthonormalised
    once at the end, so the fit is never fighting a constraint during it.
    """
    x, target = stacked(pos, neg)
    model = Subspace(leading_directions(pos, neg, k)).to(x.device)
    fit_params(model, x, target, F.binary_cross_entropy_with_logits, **opts)

    return model.orthonormal()


def manifold_probe(pos, neg, n, activation=F.gelu, **opts):
    """n activated directions: weight (n, d) and bias (n,).

    Each unit starts as a half-space pointing from the negative bulk at one spread-out positive,
    with its boundary at the midpoint. ReLU ties the region to a polytope; GELU takes the corners
    off. So the region begins as a union of half-spaces around real examples.
    """
    x, target = stacked(pos, neg)
    middle = neg.mean(0)
    points = spread_points(pos, n)

    weight = normalize_rows(points - middle)
    bias = -(weight * (points + middle) / 2).sum(1)

    model = Manifold(weight, bias, activation).to(x.device)
    fit_params(model, x, target, F.binary_cross_entropy_with_logits, **opts)

    return model.units.weight.detach(), model.units.bias.detach()


def normalize_rows(vs):
    """Each row to unit norm."""
    return vs / vs.norm(dim=1, keepdim=True).clamp(min=EPS)
