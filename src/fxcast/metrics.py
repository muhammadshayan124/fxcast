"""Accuracy metrics and the significance tests that decide whether a difference is real.

The central tool is the Diebold-Mariano test of equal predictive accuracy. Two details matter
and are easy to get wrong:

* h-step forecasts of overlapping h-day returns have errors that are autocorrelated (an MA(h-1)
  process even for an optimal forecast). A naive t-test on the loss differences would treat
  ~h overlapping observations as independent and overstate significance. The variance of the
  mean loss differential uses a Bartlett-weighted long-run variance with h-1 lags.
* With a few thousand forecasts the asymptotic normal approximation is slightly liberal, so the
  Harvey-Leybourne-Newbold small-sample correction is applied and Student's t is used.

A table with 18 model-currency-horizon comparisons will contain a "significant" one by chance
alone at the 5% level, so p-values are adjusted for multiplicity (Holm) before anything is called
a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err)))


def r2_oos(err_model: np.ndarray, err_bench: np.ndarray) -> float:
    """Out-of-sample R^2 against a benchmark (Campbell-Thompson): 1 - MSE_model / MSE_benchmark.
    Positive means the model beat the benchmark; it can be arbitrarily negative."""
    return float(1.0 - np.mean(np.square(err_model)) / np.mean(np.square(err_bench)))


@dataclass(frozen=True)
class DMResult:
    mean_diff: float  # mean(loss_a - loss_b); positive = a is worse than b
    statistic: float
    p_value: float  # two-sided
    n: int


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> DMResult:
    """Diebold-Mariano test of H0: E[loss_a] == E[loss_b], two-sided, with the HLN correction."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = d.size
    if n < 10:
        raise ValueError("need at least 10 forecasts for a DM test")
    mean = d.mean()
    dc = d - mean
    lrv = float(dc @ dc) / n  # gamma_0
    for k in range(1, h):  # Bartlett weights keep the estimate non-negative
        gamma = float(dc[k:] @ dc[:-k]) / n
        lrv += 2.0 * (1.0 - k / h) * gamma
    if lrv <= 0:  # degenerate (identical forecasts): no evidence either way
        return DMResult(float(mean), 0.0, 1.0, n)
    stat = mean / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= hln
    p = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return DMResult(float(mean), float(stat), float(p), n)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def directional(forecast: np.ndarray, actual: np.ndarray, step: int = 1) -> tuple[float, float, int]:
    """Hit rate of the forecast's sign, with a two-sided binomial test against 50%.

    Subsampled every `step` origins: with h-day overlapping returns, consecutive hits are
    dependent, and testing all of them would overstate significance. Zero forecasts or zero
    outcomes carry no direction and are excluded.
    """
    f, a = np.asarray(forecast)[::step], np.asarray(actual)[::step]
    keep = (f != 0) & (a != 0)
    f, a = f[keep], a[keep]
    if f.size == 0:
        return float("nan"), 1.0, 0
    hits = int(np.sum(np.sign(f) == np.sign(a)))
    return hits / f.size, float(stats.binomtest(hits, f.size, 0.5).pvalue), int(f.size)


def qlike(forecast_var: np.ndarray, proxy_var: np.ndarray) -> np.ndarray:
    """QLIKE loss for variance forecasts (up to a constant that cancels in loss differences).

    Patton (2011): among common losses it is robust to noise in the variance proxy, which is
    essential here because a squared daily return is an extremely noisy measure of variance.
    Penalises under-prediction of variance more than over-prediction.
    """
    fc = np.maximum(np.asarray(forecast_var, dtype=float), 1e-14)
    return np.log(fc) + np.asarray(proxy_var, dtype=float) / fc
