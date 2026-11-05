"""Collection of utils and tool to extract subspaces.

Flip a concept and the activation moves; that displacement is where it is stored, and removing it
is what repairing a shortcut means. A discriminant -- a separating hyperplane's normal -- divides
that displacement by the covariance, so it prefers a quiet axis that separates well over the loud
one carrying the content. Both are here because reaching for the second is the mistake.
"""
import torch

from reliance.layout import EPS, gauge, normalize, orient
# `svm` is re-exported, not used here: every estimator stays reachable from one module
from reliance.geometric_probes import linear_probe, svm


def orthobasis(vs):
    """Orthonormal basis (k, d) for the span of `vs`. QR in double; a span has no sign."""
    q, _ = torch.linalg.qr(vs.T.double())

    return q.T.float()


def angle(a, b):
    """Angle between two directions in degrees, ignoring sign. How wrong an estimate is.

    Via the chord, 2 asin(||a - b|| / 2), rather than arccos of the dot product: arccos is
    ill-conditioned exactly where it matters most, returning hundredths of a degree of noise for
    two vectors that are equal in float32.
    """
    a = a.flatten() / a.norm().clamp(min=EPS)
    b = b.flatten() / b.norm().clamp(min=EPS)
    b = -b if a @ b < 0 else b

    return float(torch.rad2deg(2 * torch.arcsin(((a - b).norm() / 2).clamp(max=1.0))))


def projector_norm(V, v):
    """||P_V v|| / ||v|| for orthonormal V: the cosine between v and the nearest point of span(V).

    1 means the subspace contains v, 0 that it misses it entirely. v is normalised here so its
    scale is irrelevant, and V's rows being orthonormal makes P_V = V^T V and ||P_V v|| = ||V v||.
    Tracked counterfactual accuracy closely in experiments 12 and 17.
    """
    v = v.flatten()
    v = v / v.norm().clamp(min=EPS)

    return float((v @ V.T).norm())


def direction_auc(V, pos, neg):
    """Share of pairs whose coordinate along `V` is larger in `pos`. Mann-Whitney AUC, 0.5 chance.

    Detection accuracy, threshold-free and exactly paired. One direction only: a subspace has no
    sign, so summing its coordinates would read an orientation `orthobasis` chose arbitrarily.
    """
    if len(V) != 1:
        raise ValueError('direction_auc takes one direction; project V out and fit a fresh probe')

    a, b = pos @ V[0], neg @ V[0]
    n = min(len(a), len(b))

    return float((a[:n] > b[:n]).float().mean())


def dom(pos, neg, k=1):
    """The difference of means, normalised. The displacement, estimated the obvious way.

    With paired rows this is also the mean of the paired differences: the mean is linear, so
    pairing buys nothing at k = 1. What it buys past the first direction is `paired_svd`.
    """
    if k != 1:
        raise ValueError('dom returns one direction; use paired_svd or pls for k > 1')

    return normalize(pos.mean(0) - neg.mean(0))


def paired_svd(pos, neg, k=1, center=False):
    """Top-k principal directions of the paired differences pos[i] - neg[i]. Rows must correspond.

    The second moment of the differences is not recoverable from two group means, which is how this
    sees past the first direction: k = 8 was the best repair in the repo, 0.957 to dom's 0.950.
    Do not centre -- the mean of the differences *is* the difference of means, so centring subtracts
    the answer (0.925 -> 0.550). The flag is there to measure the mistake, not to use it.
    """
    diff = pos - neg

    if center:
        diff = diff - diff.mean(0)

    _, _, vh = torch.linalg.svd(diff.double(), full_matrices=False)

    return orient(vh[:k].float(), diff.mean(0))


def dom_svd(pos, neg, k=1):
    """The difference of means as atom 0, then the top k - 1 principal directions of what is left.

    `paired_svd`'s leading singular vector only approximates the mean: where the differences vary
    more along a direction than they shift along it, the mean is spread across several atoms and
    none of them is it. Here the mean is taken out first and held, the centred differences are
    projected off it before the SVD, and the residual principal directions fill the remaining
    rank -- so the span always contains the displacement and the extra atoms buy only what the
    mean missed. On pseudo-paired differences the two spans covered the displacement equally and
    this one repaired better on every arm tried, by 0.01 to 0.11 of worst-group accuracy. Rows
    must correspond; double throughout; k = 1 is `dom`.
    """
    diff = (pos - neg).double()

    if k > min(len(diff), diff.shape[1]):
        raise ValueError(f'asked for {k} directions from {len(diff)} differences '
                         f'in {diff.shape[1]} dimensions')

    mean = diff.mean(0)
    u = mean / mean.norm().clamp(min=EPS)
    centred = diff - mean
    residual = centred - torch.outer(centred @ u, u)
    _, s, vh = torch.linalg.svd(residual, full_matrices=False)

    # a slot filled from the nullspace would be an arbitrary direction, so a set of differences
    # that does not vary in k - 1 directions past its mean refuses rather than pads
    rank = int((s > max(residual.shape) * torch.finfo(residual.dtype).eps * s[0]).sum())

    if rank < k - 1:
        raise ValueError(f'the differences vary in {rank} directions past their mean; asked for {k - 1}')

    rest = vh[:k - 1]
    rest = rest - torch.outer(rest @ u, u)
    rest = rest / rest.norm(dim=1, keepdim=True).clamp(min=EPS)

    return torch.cat([u[None], rest]).float()


def pls(pos, neg, k=1):
    """Partial least squares: directions of maximum covariance with the concept, needing no pairs.

    w_1 is proportional to X^T y, which for two balanced groups is exactly `dom`. Past that it adds
    little on a toy -- covariance with y is spent once the mean shift is deflated, so a cone whose
    spread is variance rather than mean is invisible to it (0.73 captured against `paired_svd`'s
    1.00). Kept because a real tap may carry mean structure a synthetic cone does not.
    """
    x = torch.cat([pos, neg]).double()
    y = torch.cat([torch.ones(len(pos)), -torch.ones(len(neg))]).to(x.device).double()
    x, y = x - x.mean(0), y - y.mean()
    found = []

    for _ in range(k):
        w = x.T @ y
        w = w / w.norm().clamp(min=EPS)
        t = x @ w

        x = x - torch.outer(t, x.T @ t / (t @ t).clamp(min=EPS))
        y = y - t * (y @ t / (t @ t).clamp(min=EPS))
        found.append(w)

    return orient(orthobasis(torch.stack(found).float()), pos.mean(0) - neg.mean(0))


def sparse_delta(pos, neg, atoms=10, active=2, steps=1000, lr=1e-2):
    """The paired differences coded over a tied dictionary, `active` atoms per difference.

    Top-k and not L1: L1 charges for a loading's size, not the atoms carrying it, so duplicates
    never merge, while a top-k slot is a budget a duplicate spends. Tied, so `topk(delta @ D.T)`
    codes an activation it never saw. Atoms are recoverable only while their mutual coherence
    stays under roughly 1 / (2 * active - 1); past that an SVD of the same data does better.
    """
    diff = gauge((pos - neg).double())

    if len(diff) < atoms:
        raise ValueError(f'{atoms} atoms wanted from {len(diff)} differences')

    start = diff[::len(diff) // atoms][:atoms].clone()
    D = (start / start.norm(dim=1, keepdim=True).clamp(min=EPS)).requires_grad_(True)
    optimiser = torch.optim.Adam([D], lr=lr)

    for _ in range(steps):
        z = encode(diff, D, active)
        loss = (z @ D - diff).pow(2).sum(-1).mean()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            D /= D.norm(dim=1, keepdim=True).clamp(min=EPS)

    with torch.no_grad():
        z = encode(diff, D, active)

    order = z.pow(2).sum(0).argsort(descending=True)

    return D.detach().float()[order], z.float()[:, order]


def encode(x, D, active):
    """The `active` largest coordinates of x in the dictionary, signed, the rest zero."""
    raw = x @ D.T
    kept = raw.abs().topk(active, dim=-1).indices

    return torch.zeros_like(raw).scatter(-1, kept, raw.gather(-1, kept))


def stratified_dom(strata, k=1):
    """dom within each level of a confound, averaged. `strata` is an iterable of (pos, neg).

    A plain dom picks up the confound in proportion to the correlation: at rho = 0.8 it carries
    0.6 of the label's direction, 31 degrees of error. Levels weighted equally, not by count.
    """
    if k != 1:
        raise ValueError('stratified_dom returns one direction')

    gaps = [pos.mean(0) - neg.mean(0) for pos, neg in strata]

    return normalize(torch.stack(gaps).mean(0))


def adjusted_dom(x, on, y, k=1):
    """dom with the label regressed out: the coefficient on `on` in the fit of x against [1, on, y].

    The mean move per unit of concept *holding y fixed*. By Frisch-Waugh-Lovell this is dom taken
    after both x and `on` have had y projected out, so it is `stratified_dom` for a y that is
    continuous or too finely divided for cells. Exact for x = on * u + y * w + noise however
    strongly the two correlate, up to collinearity. Intercept fitted and discarded.
    """
    if k != 1:
        raise ValueError('adjusted_dom returns one direction')

    design = torch.stack([torch.ones_like(on).float(), on.float(), y.float()], dim=1)

    return normalize(torch.linalg.lstsq(design.double(), x.double()).solution[1].float())


def fisher(pos, neg, k=1, shrink=1e-2):
    """Sigma^-1 times the displacement: the gap, divided by how the features covary.

    The textbook discriminant and the worst estimator in experiment 17, at 0.541. Sigma^-1 discounts
    high-variance directions, which is backwards when high variance is what makes one worth
    removing. Shrinkage is not optional at these widths; quoted against the mean diagonal.
    """
    if k != 1:
        raise ValueError('fisher returns one direction')

    mu = (pos.mean(0) - neg.mean(0)).double()
    within = torch.cat([pos - pos.mean(0), neg - neg.mean(0)]).double()

    cov = within.T @ within / len(within)
    ridge = cov + shrink * cov.diagonal().mean() * torch.eye(len(cov), device=cov.device,
                                                             dtype=cov.dtype)

    return normalize(torch.linalg.solve(ridge, mu).float())


def inlp(pos, neg, k=8, **opts):
    """Iterated nullspace projection: fit a probe, remove it, refit. k times.

    The only estimator here that optimises for the concept being undetectable rather than being
    scored on it afterwards. Each round trains on data the previous directions were projected out
    of, so they come out orthogonal for free. It does not work yet: see test_contrast_subspaces.py.
    """
    found = []

    for _ in range(k):
        v = linear_probe(pos, neg, **opts)[0]
        found.append(v)
        pos = pos - (pos @ v)[:, None] * v
        neg = neg - (neg @ v)[:, None] * v

    return orthobasis(torch.stack(found))
