"""The toy: a sparse overcomplete concept code read through a random dictionary.

`c` concepts as unit directions in R^d, `l0` firing per sample. Direction 0 is the core and
direction 1 is the spurious; both fire half the time and always together, and are no louder than
any other concept. `snr` is gaussian grain as a fraction of ||x||; x then goes on the sphere of
radius `scale`, which is the only magnitude knob. That last step is not cosmetic: two concepts
firing together half the time would otherwise make ||x|| jump, and the label would read straight
off the norm -- a third rule nothing here measures, and one a model will happily take instead.
"""
import torch
from einops import rearrange

CORE, SPUR, FILLER = 0, 1, 2

# the setups tested. `margin` is d / (l0 log(c / l0)), the room over the sparse-recovery budget.
# Margin > 1 is enough to make the core linearly readable, but not enough to read it with the
# spurious gone and then transfer to a distribution never trained on -- that wants about 2 or more,
# and below it the benchmark measures the recovery deficit rather than the method.
SETUPS = {'small': dict(d=32, l0=4), 'medium': dict(d=128, l0=8), 'large': dict(d=512, l0=16)}
SCALES = (1.0, 10.0, 100.0)
DEPTHS = (1, 3)


class Concepts:
    """x = (sparse code) @ D + grain, put on the sphere of radius `scale`."""

    def __init__(self, d, l0=0.01, snr=0.1, scale=1.0, c=None):
        self.d = d
        self.c = c or 10 * d
        self.l0 = max(2, round(l0 * self.c) if l0 < 1 else int(l0))
        self.snr, self.scale = snr, scale

        D = torch.randn(self.c, self.d)
        self.D = D / D.norm(dim=1, keepdim=True)

    def sample(self, n, core=None, spur=None):
        """n vectors of size d, plus the label. `core`/`spur` force a concept on or off."""
        y = torch.rand(n) < 0.5 if core is None else torch.full((n,), bool(core))
        on = y if spur is None else torch.full((n,), bool(spur))

        z = torch.zeros(n, self.c)
        loose = torch.rand(n, self.c - FILLER)
        z.scatter_(1, loose.topk(self.l0 - 2, 1, largest=False).indices + FILLER,
                   self.amplitude((n, self.l0 - 2)))
        z[:, CORE] = y.float() * self.amplitude(n)
        z[:, SPUR] = on.float() * self.amplitude(n)

        x = z @ self.D
        x = x + self.snr * x.norm(dim=1, keepdim=True) * torch.randn(n, self.d) / self.d ** 0.5
        x = self.scale * x / x.norm(dim=1, keepdim=True)

        return rearrange(x, 'n d -> n 1 d'), y.float()

    def amplitude(self, shape):
        """How loud a firing concept is: uniform on [1, 2], always positive."""
        shape = shape if isinstance(shape, tuple) else (shape,)

        return 1.0 + torch.rand(*shape)

    def counterfactual(self, n):
        """Half core without the spurious, half spurious without the core. They disagree on all."""
        honest, ones = self.sample(n // 2, core=True, spur=False)
        cheat, zeros = self.sample(n // 2, core=False, spur=True)

        return torch.cat([honest, cheat]), torch.cat([ones, zeros])

    def get_core(self):
        return self.D[CORE]

    def get_spurious(self):
        return self.D[SPUR]
