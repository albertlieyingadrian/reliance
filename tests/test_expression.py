"""Closed-form checks for expression.py."""
import torch
import pytest
from reliance import expression as E


def test_score_direction():
    """score(z, v) is the dot product <z, v> when z is a single activation."""
    z = torch.tensor([1.0, 2.0, 3.0])
    v = torch.tensor([0.0, 1.0, 0.0])
    assert E.score(z, v).item() == pytest.approx(2.0)


def test_score_batch():
    """score broadcasts correctly over a batch of activations."""
    z = torch.ones(4, 3)
    v = torch.tensor([1.0, 0.0, 0.0])
    out = E.score(z, v)
    assert out.shape == (4,)
    assert out.allclose(torch.ones(4))


def test_score_spatial():
    """score broadcasts over (n, p, c) activations."""
    z = torch.ones(2, 7, 3)
    v = torch.tensor([0.0, 0.0, 1.0])
    out = E.score(z, v)
    assert out.shape == (2, 7)


def test_expression_cost_uniform():
    """With gamma=1 and all scores equal, expression_cost equals that score."""
    scores = torch.full((5,), 2.0)
    assert E.expression_cost(scores, gamma=1.0, reduce='mean').item() == pytest.approx(2.0)


def test_expression_cost_discount():
    """Discounting later steps: cost with gamma<1 is strictly less than the undiscounted mean
    when scores are constant and gamma < 1."""
    scores = torch.ones(4)
    full  = E.expression_cost(scores, gamma=1.0, reduce='mean').item()
    discounted = E.expression_cost(scores, gamma=0.5, reduce='mean').item()
    assert discounted < full


def test_expression_cost_max():
    """'max' reduce returns the highest discounted score."""
    scores = torch.tensor([0.5, 3.0, 1.0])
    # with gamma=1 all weights are 1, so max = 3.0
    assert E.expression_cost(scores, gamma=1.0, reduce='max').item() == pytest.approx(3.0)


def test_expression_cost_relu():
    """Negative scores do not contribute: the cost uses relu(score)."""
    scores = torch.tensor([-5.0, -1.0, 0.0])
    assert E.expression_cost(scores, gamma=1.0, reduce='mean').item() == pytest.approx(0.0)


def test_expression_cost_batched():
    """Batched input (n, T) returns one cost per rollout."""
    scores = torch.ones(3, 5)
    out = E.expression_cost(scores, gamma=1.0, reduce='mean')
    assert out.shape == (3,)
    assert out.allclose(torch.ones(3))
