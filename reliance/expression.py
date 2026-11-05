"""Expression shaping core cost function.

A concept estimator (such as a linear probe, or the projection onto a direction v) gives a
scalar score for each activation.  Expression shaping turns that score into a training cost:
rollouts that express the unwanted concept more are assigned a higher cost, so the policy
is updated to produce them less.

This module handles the aggregation step -- turning per-step scores into one cost per rollout --
and the two ways to attach that cost to training (see Section 1 of the paper).
"""
import torch


def score(z, v):
    """The concept score at one or several positions: the projection of z onto v.

    z can be a single activation (c,), a batch (n, c), or a spatial batch (n, p, c).
    v is a unit direction (c,).  The return shape matches z with the last axis dropped.
    """
    return z @ v


def expression_cost(scores, gamma=1.0, reduce='mean'):
    """Turn per-step concept scores into one cost per rollout.

    scores: (T,) or (n, T) -- one score per generation step, optionally batched over rollouts.
    gamma:  discount factor applied to later steps.  gamma=1 weights every step equally;
            smaller values make the cost track the earliest occurrence of the concept.
    reduce: 'mean' averages discounted scores; 'max' takes the strongest occurrence.
            Bergen et al. (2026) show that 'max' makes it harder for the policy to evade
            the estimator, because the policy has to suppress the concept everywhere.

    Returns a scalar (or (n,) if the input was batched).

    This implements Eq. (2) from the paper with rho = relu (nonneg cost), so only steps
    where the concept fires contribute to the penalty.
    """
    batched = scores.dim() == 2
    if not batched:
        scores = scores.unsqueeze(0)

    T = scores.shape[-1]
    weights = gamma ** torch.arange(T, dtype=scores.dtype, device=scores.device)
    weighted = weights * scores.clamp(min=0.0)

    if reduce == 'mean':
        result = weighted.mean(dim=-1)
    elif reduce == 'max':
        result = weighted.max(dim=-1).values
    else:
        raise ValueError(f"reduce must be 'mean' or 'max', got {reduce!r}")

    return result if batched else result.squeeze(0)
