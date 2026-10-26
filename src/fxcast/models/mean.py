"""Forecasters of the h-day-ahead cumulative log return R_{t,h}.

Ordered from the benchmark upward in flexibility. Hyperparameters are fixed a priori and are not
tuned on the evaluation period: tuning on the test window is the quiet way a backtest becomes
an in-sample fit. Ridge selects its penalty by time-series cross-validation *inside* each
training window only.

All models see targets scaled to percent (x100) so regularisation constants are not at the
mercy of returns being ~1e-3.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCALE = 100.0
AR_COLUMNS = [f"ret_lag{k}" for k in range(5)]


class MeanModel(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series, h: int) -> None: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class RandomWalk:
    """The benchmark: the best forecast of the future log rate is the current one (zero return).
    Meese & Rogoff (1983) found that structural exchange-rate models could not beat it out of
    sample, and a great deal of later work has not overturned that for short horizons."""

    name = "rw"

    def fit(self, X, y, h):
        pass

    def predict(self, X):
        return np.zeros(len(X))


class Drift:
    """Random walk plus the historical average daily return, scaled to the horizon."""

    name = "drift"

    def fit(self, X, y, h):
        self.mu = float(np.mean(y.to_numpy() / h))
        self.h = h

    def predict(self, X):
        return np.full(len(X), self.mu * self.h)


class AR5:
    """Direct linear regression of the h-day return on the last five daily returns."""

    name = "ar5"

    def fit(self, X, y, h):
        self.m = LinearRegression().fit(X[AR_COLUMNS].to_numpy(), y.to_numpy() * SCALE)

    def predict(self, X):
        return self.m.predict(X[AR_COLUMNS].to_numpy()) / SCALE


class Ridge:
    """All features, standardised, with the ridge penalty chosen by time-series CV inside the training window."""

    name = "ridge"

    def fit(self, X, y, h):
        self.m = make_pipeline(
            StandardScaler(), SimpleImputer(strategy="constant", fill_value=0.0),
            RidgeCV(alphas=np.logspace(-1, 5, 13), cv=TimeSeriesSplit(5)),
        ).fit(X.to_numpy(), y.to_numpy() * SCALE)

    def predict(self, X):
        return self.m.predict(X.to_numpy()) / SCALE


class HGB:
    """Gradient-boosted trees, deliberately small and heavily regularised: shallow, slow, large leaves."""

    name = "hgb"

    def fit(self, X, y, h):
        self.m = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=False, random_state=0,
        ).fit(X.to_numpy(), y.to_numpy() * SCALE)

    def predict(self, X):
        return self.m.predict(X.to_numpy()) / SCALE


def default_factories() -> dict[str, Callable[[], MeanModel]]:
    return {"rw": RandomWalk, "drift": Drift, "ar5": AR5, "ridge": Ridge, "hgb": HGB}


BENCHMARK = "rw"
