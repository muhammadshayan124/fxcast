"""Strictly causal features and targets.

Every feature at row t uses information available at the close of day t and nothing later.
That is enforced two ways: by construction (only trailing, right-aligned windows), and by a
test that edits the future of a series and asserts the past features do not move.

Targets are *forward* quantities and are the only thing here that looks ahead; they exist only
to train and score, and the walk-forward engine purges them (see `splits.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RET_LAGS = (0, 1, 2, 3, 4)  # r_t, r_{t-1}, ... r_{t-4}
CUM_WINDOWS = (5, 10, 20, 60, 120)  # trailing cumulative log return
VOL_WINDOWS = (5, 20, 60)  # trailing realised volatility
MA_WINDOW = 200


def own_features(y: pd.Series) -> pd.DataFrame:
    """Features of one currency from its own log-level series `y` (date-indexed, no NaNs)."""
    r = y.diff()
    cols: dict[str, pd.Series] = {}
    for k in RET_LAGS:
        cols[f"ret_lag{k}"] = r.shift(k)
    for w in CUM_WINDOWS:
        cols[f"cum_{w}"] = y - y.shift(w)
    for w in VOL_WINDOWS:
        cols[f"vol_{w}"] = r.rolling(w).std()
    cols[f"gap_ma{MA_WINDOW}"] = y - y.rolling(MA_WINDOW).mean()
    return pd.DataFrame(cols)


def cross_features(levels: pd.DataFrame, target: str) -> pd.DataFrame:
    """Trailing 5- and 20-day returns of the *other* currencies. Same-date alignment is safe: all
    fixings are published at the same time on the same day."""
    out = {}
    for c in levels.columns:
        if c == target:
            continue
        lv = levels[c].ffill()
        out[f"x_{c}_5"] = lv - lv.shift(5)
        out[f"x_{c}_20"] = lv - lv.shift(20)
    return pd.DataFrame(out, index=levels.index)


def build_features(levels: pd.DataFrame, target: str) -> pd.DataFrame:
    y = levels[target].dropna()
    base = own_features(y)
    cross = cross_features(levels, target).reindex(base.index)
    return pd.concat([base, cross], axis=1)


def forward_return(y: pd.Series, h: int) -> pd.Series:
    """R_{t,h} = y_{t+h} - y_t: the h-day-ahead cumulative log return, indexed by the *origin* t."""
    return y.shift(-h) - y


def forward_variance(r: pd.Series, h: int) -> pd.Series:
    """Per-day realised variance over the next h days, indexed by the origin t:
    mean(r_{t+1}^2, ..., r_{t+h}^2). A noisy but unbiased proxy for the latent variance."""
    sq = r**2
    fwd = sum(sq.shift(-k) for k in range(1, h + 1)) / h
    return fwd


def trailing_sq(r: pd.Series, windows: tuple[int, ...] = (1, 5, 22, 66)) -> pd.DataFrame:
    """Trailing mean squared returns, the building block of HAR-type volatility regressions."""
    sq = r**2
    return pd.DataFrame({f"rv_{w}": sq.rolling(w).mean() for w in windows})


def valid_rows(frame: pd.DataFrame) -> np.ndarray:
    return ~frame.isna().any(axis=1).to_numpy()
