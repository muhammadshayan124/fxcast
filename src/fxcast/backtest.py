"""Walk-forward backtests for the mean track and the volatility track, and their summaries."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fxcast import metrics
from fxcast.features import build_features, forward_return, forward_variance, own_features, trailing_sq
from fxcast.models import mean as mean_models
from fxcast.models import vol as vol_models
from fxcast.splits import blocks, train_end

MIN_TRAIN = 1000  # about four years of daily data before the first forecast
REFIT_EVERY = 63  # about one quarter
HORIZONS = (1, 5, 20)


def _first_valid(frame: pd.DataFrame) -> int:
    ok = frame.notna().all(axis=1).to_numpy()
    if not ok.any():
        raise ValueError("no row has all features available")
    return int(np.argmax(ok))


def _check_purged(origin: int, last_train: int, h: int, labels: pd.Series) -> None:
    """Every training label must be fully realised at the forecast origin: the label of row s needs
    price s+h, so s + h <= origin. NaN labels mean the same thing from the other side."""
    if last_train + h > origin or labels.isna().any():
        raise AssertionError("training labels are not all realised at the forecast origin")


def mean_predictions(
    levels: pd.DataFrame, currency: str, h: int, factories: dict[str, Callable] | None = None,
    min_train: int = MIN_TRAIN, refit_every: int = REFIT_EVERY,
) -> pd.DataFrame:
    """Walk-forward forecasts of R_{t,h}. Long format: one row per (origin, model)."""
    factories = factories or mean_models.default_factories()
    y = levels[currency].dropna()
    X = build_features(levels, currency).reindex(y.index)
    target = forward_return(y, h)
    first_valid = _first_valid(own_features(y))
    first_origin = first_valid + min_train

    rows = []
    for b in blocks(first_origin, len(y), h, refit_every):
        tr = slice(first_valid, train_end(b.start, h) + 1)
        X_tr, y_tr = X.iloc[tr], target.iloc[tr]
        _check_purged(b.start, train_end(b.start, h), h, y_tr)
        X_te = X.iloc[b.start : b.end]
        actual = target.iloc[b.start : b.end].to_numpy()
        for name, make in factories.items():
            model = make()
            model.fit(X_tr, y_tr, h)
            rows.append(pd.DataFrame({
                "date": X_te.index, "currency": currency, "h": h, "model": name,
                "forecast": model.predict(X_te), "actual": actual,
            }))
    return pd.concat(rows, ignore_index=True)


def vol_predictions(
    levels: pd.DataFrame, currency: str, h: int, min_train: int = MIN_TRAIN, refit_every: int = REFIT_EVERY,
) -> tuple[pd.DataFrame, int]:
    """Walk-forward forecasts of average daily variance over the next h days.
    Returns (predictions, number of GARCH blocks that failed to fit and fell back to EWMA)."""
    y = levels[currency].dropna()
    r = y.diff()
    target = forward_variance(r, h)
    feats = trailing_sq(r)
    expanding, ewma = vol_models.expanding_variance(r), vol_models.ewma_variance(r)
    first_valid = _first_valid(feats)
    first_origin = first_valid + min_train

    rows, fallbacks = [], 0
    for b in blocks(first_origin, len(y), h, refit_every):
        tr = slice(first_valid, train_end(b.start, h) + 1)
        X_tr, y_tr = feats.iloc[tr], target.iloc[tr]
        _check_purged(b.start, train_end(b.start, h), h, y_tr)
        idx = y.index[b.start : b.end]
        actual = target.iloc[b.start : b.end].to_numpy()
        garch = vol_models.garch_block_forecast(r, b.start, b.start, b.end, h)
        if garch is None:
            fallbacks += 1
            garch = ewma.iloc[b.start : b.end].to_numpy()
        forecasts = {
            "expanding": expanding.iloc[b.start : b.end].to_numpy(),
            "ewma": ewma.iloc[b.start : b.end].to_numpy(),
            "har": vol_models.har_fit_predict(X_tr, y_tr, feats.iloc[b.start : b.end]),
            "garch": garch,
        }
        for name, fc in forecasts.items():
            rows.append(pd.DataFrame({"date": idx, "currency": currency, "h": h, "model": name, "forecast": fc, "actual": actual}))
    return pd.concat(rows, ignore_index=True), fallbacks


def run_mean(levels: pd.DataFrame, currencies, horizons=HORIZONS, n_jobs: int = 4, **kw) -> pd.DataFrame:
    jobs = [(c, h) for c in currencies for h in horizons]
    parts = Parallel(n_jobs=n_jobs)(delayed(mean_predictions)(levels, c, h, **kw) for c, h in jobs)
    return pd.concat(parts, ignore_index=True)


def run_vol(levels: pd.DataFrame, currencies, horizons=HORIZONS, n_jobs: int = 4, **kw) -> tuple[pd.DataFrame, int]:
    jobs = [(c, h) for c in currencies for h in horizons]
    parts = Parallel(n_jobs=n_jobs)(delayed(vol_predictions)(levels, c, h, **kw) for c, h in jobs)
    return pd.concat([p for p, _ in parts], ignore_index=True), sum(f for _, f in parts)


# -- summaries ---------------------------------------------------------------------------------

def summarize_mean(preds: pd.DataFrame, bench: str = mean_models.BENCHMARK) -> pd.DataFrame:
    """One row per (currency, horizon, model != benchmark). DM p-values are Holm-adjusted across
    every comparison in the table, because that is the number of looks we took."""
    rows = []
    for (cur, h), g in preds.groupby(["currency", "h"]):
        wide = g.pivot(index="date", columns="model", values="forecast")
        actual = g.groupby("date")["actual"].first().loc[wide.index].to_numpy()
        e_b = actual - wide[bench].to_numpy()
        for model in wide.columns:
            if model == bench:
                continue
            e_m = actual - wide[model].to_numpy()
            dm = metrics.dm_test(e_m**2, e_b**2, h=int(h))
            acc, acc_p, acc_n = metrics.directional(wide[model].to_numpy(), actual, step=int(h))
            rows.append({
                "currency": cur, "h": int(h), "model": model, "n": dm.n,
                "rmse_bp": metrics.rmse(e_m) * 1e4, "rmse_bench_bp": metrics.rmse(e_b) * 1e4,
                "rel_rmse": metrics.rmse(e_m) / metrics.rmse(e_b), "r2_oos": metrics.r2_oos(e_m, e_b),
                "dm_stat": dm.statistic, "dm_p": dm.p_value,
                "dir_acc": acc, "dir_p": acc_p, "dir_n": acc_n,
            })
    out = pd.DataFrame(rows)
    out["dm_p_holm"] = metrics.holm_adjust(out["dm_p"].tolist())
    out["dir_p_holm"] = metrics.holm_adjust(out["dir_p"].tolist())
    return out.sort_values(["h", "currency", "model"]).reset_index(drop=True)


def summarize_vol(preds: pd.DataFrame, bench: str = vol_models.BENCHMARK) -> pd.DataFrame:
    rows = []
    for (cur, h), g in preds.groupby(["currency", "h"]):
        wide = g.pivot(index="date", columns="model", values="forecast")
        actual = g.groupby("date")["actual"].first().loc[wide.index].to_numpy()
        loss_b = metrics.qlike(wide[bench].to_numpy(), actual)
        for model in wide.columns:
            if model == bench:
                continue
            loss_m = metrics.qlike(wide[model].to_numpy(), actual)
            dm = metrics.dm_test(loss_m, loss_b, h=int(h))
            rows.append({
                "currency": cur, "h": int(h), "model": model, "n": dm.n,
                "qlike": float(loss_m.mean()), "qlike_bench": float(loss_b.mean()),
                "qlike_gain": float(loss_b.mean() - loss_m.mean()),  # positive = better than the benchmark
                "dm_stat": dm.statistic, "dm_p": dm.p_value,
            })
    out = pd.DataFrame(rows)
    out["dm_p_holm"] = metrics.holm_adjust(out["dm_p"].tolist())
    return out.sort_values(["h", "currency", "model"]).reset_index(drop=True)


def economic_h1(preds: pd.DataFrame, cost: float = 1e-4) -> pd.DataFrame:
    """Does trading the sign of the 1-day forecast make money after a transaction cost?

    Position = sign(forecast) held for one day; PnL = position * next-day log return; a cost of
    `cost` (1 bp) is paid per unit of position change. Spot returns only: no interest-rate
    differential (carry), so this is a test of the forecast, not a realistic FX strategy.
    """
    rows = []
    for (cur, model), g in preds[preds["h"] == 1].groupby(["currency", "model"]):
        if model == mean_models.BENCHMARK:
            continue
        g = g.sort_values("date")
        pos = np.sign(g["forecast"].to_numpy())
        pnl_gross = pos * g["actual"].to_numpy()
        turnover = np.abs(np.diff(pos, prepend=0.0))
        pnl_net = pnl_gross - cost * turnover
        ann = np.sqrt(252)
        rows.append({
            "currency": cur, "model": model,
            "sharpe_gross": float(pnl_gross.mean() / pnl_gross.std() * ann) if pnl_gross.std() else 0.0,
            "sharpe_net": float(pnl_net.mean() / pnl_net.std() * ann) if pnl_net.std() else 0.0,
            "turnover": float(turnover.mean()),
        })
    cols = ["currency", "model", "sharpe_gross", "sharpe_net", "turnover"]
    return pd.DataFrame(rows, columns=cols).sort_values(["currency", "model"]).reset_index(drop=True)
