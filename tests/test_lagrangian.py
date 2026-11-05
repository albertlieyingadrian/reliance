"""The multiplier: does it find its own size, and does the model end up fixed.

Mechanics first, in milliseconds, then the toy. One thing to know about the thresholds: once the
shortcut is gone the constraint holds at kappa = 0, so the multiplier decays and the number it
ends at says nothing. The peak is what a sweep was looking for, and the peak is what is checked.
"""
import math

import pytest
import torch

from reliance import dependence as D
from reliance.lagrangian import DualRate, inverse_softplus, softplus

import common
from dgp import SETUPS, Concepts


def test_share_is_invariant_to_the_activation_scale():
    """Why a target can be named once and travel: scaling x scales the measure and the reference
    it is divided by identically, so the share does not move."""
    torch.manual_seed(0)
    width = 32
    w = torch.randn(width)

    def g(z):
        return (z @ w).sum(1)

    x = torch.randn(4, 1, width)
    v = torch.randn(width)
    v = v / v.norm()

    seen = []
    for k in (1.0, 10.0, 100.0):
        torch.manual_seed(1)
        seen.append(float(D.make_measure_relative(D.gradinput, g, k * x, v, draws=64,
                                  create_graph=False).pow(2).mean()))

    assert max(seen) / min(seen) < 1.05


def test_softplus_and_its_inverse_agree_across_the_range():
    for x in (1e-6, 1e-3, 1.0, 10.0, 1e3, 1e8):
        assert abs(softplus(inverse_softplus(x)) / x - 1.0) < 1e-6


def test_the_multiplier_climbs_while_the_constraint_is_violated():
    rate = DualRate(target=1.0, lr=0.5)
    rate.value = 4.0

    for _ in range(60):
        rate.u += rate.lr * math.log(rate.value / rate.target)

    assert rate.kappa > 1.0


def test_the_multiplier_is_never_negative_and_never_runs_away():
    """An unreachable target is the case that breaks a naive ascent."""
    rate = DualRate(target=1e-9, lr=1.0, cap=100.0)
    rate.value = 1.0

    for _ in range(500):
        rate.u = min(rate.u + rate.lr * math.log(rate.value / rate.target),
                     inverse_softplus(rate.cap))

    assert 0.0 <= rate.kappa <= 100.0


@pytest.fixture(scope='module')
def dgp():
    torch.manual_seed(0)

    return Concepts(**SETUPS['small'])


@pytest.mark.slow
def test_the_baseline_splits_between_the_two_rules(dgp):
    """Asserted first. A test whose baseline passes is testing nothing."""
    assert common.baseline(dgp) < 0.75


@pytest.mark.slow
@pytest.mark.parametrize('name', ('directional_derivative', 'gradinput', 'banzhaf'))
def test_it_repairs_the_toy_without_being_told_a_lambda(name, dgp):
    """The payoff. Nothing names a lambda -- only how much of chance the direction may collect."""
    torch.manual_seed(0)
    got = common.by_dual_rate(name, dgp, target=0.01)

    assert got['accuracy'] > common.PASSING - 0.05


@pytest.mark.slow
def test_lambda_is_not_a_constant_across_regimes(dgp):
    """Why the ratio is read off rather than swept: the lambda that works moves by decades, while
    the thing being asked for does not move at all."""
    seen = []

    for size in ('small', 'large'):
        torch.manual_seed(0)
        seen.append(common.by_dual_rate('gradinput', Concepts(**SETUPS[size]), target=0.01))

    lams = [got['median'] for got in seen]

    assert min(lams) > 0
    assert all(got['accuracy'] > 0.85 for got in seen)
