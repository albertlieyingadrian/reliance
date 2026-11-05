"""Facilitate the tuning of Lambda using dual rate method.

At a stationary point of L + lambda P the gradients balance, so lambda ~ ||grad L|| / ||grad P||:
an exchange rate between quantities in different units, not a hyperparameter. `DualRate` reads
that ratio off the gradients and ascends the one dimensionless factor left against a constraint,
named in units of `dependence.make_measure_relative`.
"""
import math

import torch


def softplus(u):
    """log(1 + e^u), without overflowing for large u."""
    if u > 30.0:
        return u

    return math.log1p(math.exp(u))


def inverse_softplus(x):
    """The u with softplus(u) = x."""
    if x > 30.0:
        return x

    return math.log(math.expm1(x))


def gradient_norm(grads):
    """||g||, over a whole parameter list at once."""
    return torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None))


class DualRate:
    """lambda = kappa ||grad L|| / ||grad P||, with kappa ascended against P <= target.

    The ratio gives the units -- two gradient norms in the same space, so everything cancels. The
    ascent gives the stopping: kappa climbs while the constraint is violated and decays once it is
    met, so a direction that falls out of use takes lambda to zero rather than dividing by a
    vanishing ||grad P||. Neither half works alone.

    `target` is the one parameter carrying a decision. Build the penalty with `relative=True` and
    it reads as a multiple of what a random direction collects, so the default 0.01 asks for 1% of
    that. It sets where training comes to rest, and it applies to the *squared* quantity, so 1% of
    the squared projection is a tenth of the projection -- stricter than it looks.

    `cap` is not a knob: it bounds lambda and the accumulator, so a long violation cannot wind up a
    debt. If it binds, the target is unreachable.
    """

    def __init__(self, target=0.01, lr=0.05, start=1e-3, ema=0.9, cap=1e6):
        self.target, self.lr, self.ema, self.cap = target, lr, ema, cap
        self.u = inverse_softplus(start)
        self.ratio, self.value = None, None
        self.history = []

    @property
    def kappa(self):
        """All that is left once the units are read off the gradients."""
        return min(softplus(self.u), self.cap)

    def step(self, task, penalty, params, opt):
        """One step on L + lambda P, then one ascent on kappa.

        The gradients are combined by hand rather than backpropagating their sum: two backward
        passes, not three. The ascent signal is log(P / target), dimensionless and symmetric.
        """
        on_task = torch.autograd.grad(task, params, retain_graph=True, allow_unused=True)
        on_penalty = torch.autograd.grad(penalty, params, allow_unused=True)

        seen = float(gradient_norm(on_task) / gradient_norm(on_penalty).clamp(min=1e-12))
        self.ratio = seen if self.ratio is None else self.ema * self.ratio + (1 - self.ema) * seen
        lam = min(self.kappa * self.ratio, self.cap)

        for param, on_l, on_p in zip(params, on_task, on_penalty):
            first = torch.zeros_like(param) if on_l is None else on_l
            second = torch.zeros_like(param) if on_p is None else on_p
            param.grad = first + lam * second

        opt.step()

        found = float(penalty.detach())
        self.value = found if self.value is None else self.ema * self.value + (1 - self.ema) * found
        self.u = min(self.u + self.lr * math.log(max(self.value, 1e-12) / self.target),
                     inverse_softplus(self.cap))

        self.history.append((lam, self.kappa, self.value))

        return lam

    @property
    def lam(self):
        return self.history[-1][0] if self.history else 0.0

    def satisfied(self, tolerance=0.1):
        """Is the smoothed constraint met, to within a relative tolerance?"""
        return self.value is not None and self.value <= self.target * (1.0 + tolerance)
