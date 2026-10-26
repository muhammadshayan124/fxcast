"""Forecasters of average daily variance over the next h days.

The benchmark is the expanding-sample variance: "volatility is constant, estimate it from
everything seen so far". Unlike the mean, variance is known to be strongly predictable
(volatility clustering), so a good model *should* beat this benchmark. If one does not, that is
as informative as the mean-forecasting result.

Models:
  expanding  benchmark: mean of r^2 over all history
  ewma       RiskMetrics: exponentially weighted r^2 with lambda = 0.94 (no fitting)
  har        OLS of future variance on trailing mean r^2 over 1, 5, 22, 66 days (Corsi-style)
  garch      GARCH(1,1), zero mean, normal innovations, refit each block, parameters then
             held fixed while the conditional variance is filtered forward day by day
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import LinearRegression

BENCHMARK = "expanding"
EWMA_LAMBDA = 0.94
PCT = 100.0  # returns are scaled to percent for GARCH so the optimiser is well conditioned
FLOOR = 1e-10
HAR_COLUMNS = ["rv_1", "rv_5", "rv_22", "rv_66"]


def expanding_variance(r: pd.Series) -> pd.Series:
    return (r**2).expanding().mean()


def ewma_variance(r: pd.Series) -> pd.Series:
    return (r**2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean()


def har_fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    m = LinearRegression().fit(X_train[HAR_COLUMNS].to_numpy() * 1e4, y_train.to_numpy() * 1e4)
    return np.maximum(m.predict(X_test[HAR_COLUMNS].to_numpy() * 1e4) / 1e4, FLOOR)


def garch_block_forecast(r: pd.Series, fit_end: int, start: int, end: int, h: int) -> np.ndarray | None:
    """Average h-day variance forecast for origins start..end-1.

    The parameters are estimated on returns up to and including position `fit_end` (the block's
    first origin; nothing later). The filter is then run over the block with those parameters
    fixed. Returns None if estimation fails, so the caller can fall back and record it.
    """
    data = r.iloc[1:end] * PCT  # data position p holds r position p + 1
    if data.isna().any():
        raise ValueError("returns must be gap-free inside the evaluated window")
    try:
        train = data.iloc[:fit_end]  # r positions 1..fit_end: nothing after the first origin of the block
        res = arch_model(train, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off", show_warning=False)
        fixed = arch_model(data, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fix(res.params)
        # origin r-position `start` == data position start - 1; the filter is causal, so rows for
        # origins inside the block use returns only up to that origin.
        fc = fixed.forecast(horizon=h, start=fit_end - 1, reindex=False).variance
        return fc.mean(axis=1).to_numpy()[: end - start] / PCT**2
    except Exception:  # noqa: BLE001 - optimiser failures are recorded by the caller, not hidden
        return None
