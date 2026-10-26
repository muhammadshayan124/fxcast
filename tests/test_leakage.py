"""The tests that matter most: nothing from the future may influence a forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxcast import backtest
from fxcast.features import build_features, forward_return, own_features, trailing_sq
from fxcast.models import vol as vol_models
from fxcast.splits import blocks, last_origin, train_end


def make_levels(n: int = 1400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    # volatility-clustered random walks
    cols = {}
    for c in ("AAA", "BBB", "CCC"):
        vol = 0.005 * np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.3)
        cols[c] = np.cumsum(rng.normal(0, 1, n) * vol)
    return pd.DataFrame(cols, index=idx)


def test_features_do_not_move_when_the_future_is_edited():
    lv = make_levels()
    cut = 900
    base = build_features(lv, "AAA")
    edited = lv.copy()
    edited.iloc[cut + 1 :] += np.random.default_rng(1).normal(0, 5, (len(lv) - cut - 1, 3))
    after = build_features(edited, "AAA")
    pd.testing.assert_frame_equal(base.iloc[: cut + 1], after.iloc[: cut + 1])


def test_own_features_use_only_trailing_data():
    y = make_levels()["AAA"]
    f1 = own_features(y)
    y2 = y.copy()
    y2.iloc[700:] = 0.0
    f2 = own_features(y2)
    pd.testing.assert_frame_equal(f1.iloc[:700], f2.iloc[:700])


def test_train_end_purges_unrealised_labels():
    for h in (1, 5, 20):
        for origin in (100, 250):
            te = train_end(origin, h)
            assert te + h <= origin  # the label of the last training row is known at the origin
            assert te + 1 + h > origin  # and it is the latest such row


def test_blocks_cover_every_origin_once_and_stop_before_unobserved_outcomes():
    n, h = 500, 20
    bl = blocks(100, n, h, 63)
    origins = [o for b in bl for o in b.origins]
    assert origins == list(range(100, last_origin(n, h) + 1))


def test_forward_return_uses_the_future_by_design_but_is_nan_at_the_end():
    y = make_levels()["AAA"]
    t = forward_return(y, 5)
    assert t.iloc[-5:].isna().all()
    assert t.iloc[10] == pytest.approx(y.iloc[15] - y.iloc[10])


def _truncation_invariant(fn, lv: pd.DataFrame, cut_a: int, cut_b: int, **kw):
    """Forecasts for origins < cut_a must be identical whether the data stops at cut_b (> cut_a + h)
    or is extended further with different values."""
    short = lv.iloc[:cut_b]
    long = lv.copy()
    long.iloc[cut_b:] += 3.0  # a very different future
    return fn(short, **kw), fn(long, **kw)


@pytest.mark.parametrize("h", [1, 5])
def test_mean_forecasts_do_not_depend_on_later_data(h):
    lv = make_levels(1500)
    kw = dict(currency="AAA", h=h, min_train=300, refit_every=50)
    short = backtest.mean_predictions(lv.iloc[:1200], **kw)
    longer = lv.copy()
    longer.iloc[1200:] += 3.0
    full = backtest.mean_predictions(longer, **kw)
    # a forecast is comparable once its block exists in both runs; the final partial block of the
    # short run is refit on the same data, so everything it predicted must match exactly
    key = ["date", "model"]
    a = short.set_index(key)["forecast"]
    b = full.set_index(key)["forecast"].reindex(a.index)
    # blocks are aligned on first_origin, so shared origins use identical training data
    last_complete_start = max(bl.start for bl in blocks(_first_origin(lv, kw), 1200, h, 50) if bl.end - bl.start == 50)
    cutoff = lv.index[last_complete_start]
    shared = a.index.get_level_values("date") < cutoff
    np.testing.assert_allclose(a[shared].to_numpy(), b[shared].to_numpy(), rtol=1e-9, atol=1e-12)
    assert shared.sum() > 100


def _first_origin(lv: pd.DataFrame, kw: dict) -> int:
    y = lv[kw["currency"]]
    from fxcast.backtest import _first_valid

    return _first_valid(own_features(y)) + kw["min_train"]


def test_vol_forecasts_do_not_depend_on_later_data_including_garch():
    lv = make_levels(1500)
    kw = dict(currency="AAA", h=5, min_train=300, refit_every=60)
    short, _ = backtest.vol_predictions(lv.iloc[:1200], **kw)
    longer = lv.copy()
    longer.iloc[1200:] += 3.0
    full, _ = backtest.vol_predictions(longer, **kw)
    first = _first_origin(lv, kw)
    complete = [b for b in blocks(first, 1200, 5, 60) if b.end - b.start == 60]
    cutoff = lv.index[complete[-1].start]
    for model in ("expanding", "ewma", "har", "garch"):
        a = short[short["model"] == model].set_index("date")["forecast"]
        b = full[full["model"] == model].set_index("date")["forecast"].reindex(a.index)
        shared = a.index < cutoff
        assert shared.sum() > 100
        np.testing.assert_allclose(a[shared].to_numpy(), b[shared].to_numpy(), rtol=1e-8, atol=1e-14, err_msg=model)


def test_garch_filter_at_an_origin_ignores_later_returns():
    lv = make_levels(1200)
    r = lv["AAA"].diff()
    r2 = r.copy()
    r2.iloc[900:] = 1.0  # absurd future
    a = vol_models.garch_block_forecast(r, 600, 600, 700, 5)
    b = vol_models.garch_block_forecast(r2, 600, 600, 700, 5)
    assert a is not None and b is not None
    # r2 differs from position 900; the forecasts for origins 600..699 only use r up to 699
    np.testing.assert_allclose(a, b[: len(a)], rtol=1e-9)


def test_backtest_refuses_labels_that_reach_past_the_origin(monkeypatch):
    lv = make_levels(1200)
    import fxcast.backtest as bt

    monkeypatch.setattr(bt, "train_end", lambda origin, h: origin)  # break the purge on purpose
    with pytest.raises(AssertionError):
        bt.mean_predictions(lv, "AAA", 5, min_train=300, refit_every=50)


def test_trailing_sq_is_causal():
    r = make_levels()["AAA"].diff()
    r2 = r.copy()
    r2.iloc[800:] = 5.0
    pd.testing.assert_frame_equal(trailing_sq(r).iloc[:800], trailing_sq(r2).iloc[:800])
