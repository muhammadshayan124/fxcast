"""Walk-forward evaluation geometry.

At forecast origin t we predict R_{t,h} = y_{t+h} - y_t. The label of training row s is
R_{s,h} = y_{s+h} - y_s, which is only *known* once day s+h has happened. So a model fitted at
origin t may train only on rows s with s + h <= t. Training on rows s in (t-h, t] would feed
the model labels computed from prices after the forecast origin: label leakage. It is easy to
miss because the features are perfectly causal; the leak is in the targets. With h = 20 and
daily data it is a 20-row overlap, enough to flatter a flexible model noticeably.

`train_end(t, h)` is the last usable training row for origin t. Everything else here is
bookkeeping: which origins exist, and which model fit serves which block of origins.
"""

from __future__ import annotations

from dataclasses import dataclass


def train_end(origin: int, h: int) -> int:
    """Last row index whose h-step label is fully realised at `origin` (inclusive)."""
    return origin - h


@dataclass(frozen=True)
class Block:
    start: int  # first origin served by this fit; the model is fitted with data up to `start`
    end: int  # one past the last origin served

    @property
    def origins(self) -> range:
        return range(self.start, self.end)


def last_origin(n: int, h: int) -> int:
    """Last origin whose outcome y_{t+h} is observed in a series of length n."""
    return n - 1 - h


def blocks(first_origin: int, n: int, h: int, refit_every: int) -> list[Block]:
    """Contiguous blocks of `refit_every` origins, refit at the start of each. Predictions inside a
    block use a model fitted at the block's first origin, never one fitted later."""
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    stop = last_origin(n, h) + 1
    out = []
    t = first_origin
    while t < stop:
        out.append(Block(t, min(t + refit_every, stop)))
        t += refit_every
    return out
