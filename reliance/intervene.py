"""All the methods that rewrite the activation during forward

`pps` is Grant et al. (arXiv:2604.16423) eq. 1 -- x + eps v during training, dropped at inference,
applied at every position of the chosen tap. The rest are its neighbours: randomise the direction,
permute it, or delete it (`dependence.project_out`, re-exported so the family reads as one).
Each takes (x, v) and returns the rewritten activation. Strengths are absolute, as in the
paper; `tests/common.py` quotes them as a fraction of the activation scale where that matters.
"""
import torch

# re-exported, not used here: `common.INTERVENTIONS` reaches it by name, and the
# family of four rewrites reads as one
from reliance.dependence import project_out


def pps(x, v, eps=1.5):
    """x + eps v. Training only -- the caller drops it at inference, which is the whole method.

    `eps` is absolute, as in the paper, and the shift is the same for every example. That is the
    defining property and not a detail: the argument that pps cannot touch a feature being *read*
    is that a constant shift moves both classes equally. Scale it per example and one whose norm
    is larger moves further, so if the norm carries any of the label the gap along v moves too --
    and pps looks like it works for a reason that is not pps. A caller who wants a portable eps
    passes a multiple of the activation's scale; it does not belong in here.
    """
    return x + eps * v


def noise(x, v, sigma=1.0):
    """x plus a random push along v, one draw per example, so the coordinate stops being readable.

    Absolute, like `pps`'s eps; the difference between them is a fixed push against a random one,
    a bias against a destroyed channel. The draw broadcasts over whatever axes sit between the
    example and the channel, so this reads the same on (n, d) and on (n, p, c).
    """
    draw = torch.randn(len(x), *(1,) * (x.ndim - 1), device=x.device)

    return x + sigma * draw * v


def shuffle(x, v):
    """Swap the coordinate along v between examples, so it keeps its marginal and loses its label.

    The batch is the whole distribution visible at this point in training, so the permutation is
    within it. Unlike `noise` this leaves the coordinate's distribution exactly intact -- what it
    destroys is only the association with the label.
    """
    order = torch.randperm(len(x), device=x.device)
    coord = x @ v

    return x + (coord[order] - coord)[..., None] * v
