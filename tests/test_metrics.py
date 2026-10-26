from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fxcast import backtest, data, metrics, report


def test_rmse_r2_hand_checked():
    e = np.array([1.0, -1.0, 2.0, -2.0])
    assert metrics.rmse(e) == pytest.approx(np.sqrt(2.5))
    assert metrics.mae(e) == pytest.approx(1.5)
    assert metrics.r2_oos(e, e) == pytest.approx(0.0)
    assert metrics.r2_oos(e / 2, e) == pytest.approx(0.75)


def test_dm_h1_matches_a_plain_paired_t_test_up_to_the_hln_factor():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 1, 500) ** 2, rng.normal(1.0, 1, 500) ** 2
    res = metrics.dm_test(a, b, h=1)
    d = a - b
    n = len(d)
    raw = d.mean() / (d.std(ddof=0) / np.sqrt(n))
    assert res.statistic == pytest.approx(raw * np.sqrt((n - 1) / n), rel=1e-9)


def test_dm_detects_a_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 2000)
    assert metrics.dm_test((base * 1.3) ** 2, base**2, 1).p_value < 1e-6
    assert metrics.dm_test(base**2, base**2, 1).p_value == 1.0  # degenerate


def test_dm_is_calibrated_under_the_null_with_overlapping_errors():
    """Two equally good forecasters, errors that are MA(h-1) as in overlapping h-step forecasts:
    the test should reject about 5% of the time at the 5% level, not the 20-30% a naive t-test gives."""
    h, n, reps = 5, 800, 400
    rng = np.random.default_rng(2)
    rejections = 0
    naive = 0
    for _ in range(reps):
        shocks = rng.normal(0, 1, n + h)
        ma = np.convolve(shocks, np.ones(h), mode="valid")[:n]  # overlapping sums
        other = np.convolve(rng.normal(0, 1, n + h), np.ones(h), mode="valid")[:n]
        loss_a, loss_b = ma**2, other**2
        rejections += metrics.dm_test(loss_a, loss_b, h).p_value < 0.05
        d = loss_a - loss_b
        naive += 2 * stats.t.sf(abs(d.mean() / (d.std(ddof=1) / np.sqrt(n))), n - 1) < 0.05
    assert rejections / reps < 0.10
    assert naive / reps > rejections / reps  # the naive test is the liberal one


def test_holm_matches_known_values_and_is_monotone():
    p = [0.01, 0.04, 0.03, 0.005]
    adj = metrics.holm_adjust(p)
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert all(a >= x for a, x in zip(adj, p, strict=True))


def test_directional_accuracy_and_zero_handling():
    f = np.array([1, -1, 1, 0, 1.0])
    a = np.array([1, -1, -1, 1, 1.0])
    acc, p, n = metrics.directional(f, a)
    assert (acc, n) == (0.75, 4)
    assert 0 < p <= 1
    assert metrics.directional(np.zeros(3), np.ones(3))[2] == 0


def test_qlike_is_minimised_at_the_true_variance():
    proxy = np.full(5, 2.0)
    grid = np.linspace(0.5, 4, 200)
    losses = [metrics.qlike(np.full(5, g), proxy).mean() for g in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(2.0, abs=0.03)


def test_economic_h1_pays_costs_and_ignores_the_benchmark():
    idx = pd.date_range("2020-01-01", periods=6)
    rows = []
    for m, fc in (("rw", [0] * 6), ("ar5", [1, -1, 1, -1, 1, -1])):
        for d, f in zip(idx, fc, strict=True):
            rows.append({"date": d, "currency": "USD", "h": 1, "model": m, "forecast": float(f), "actual": 0.001})
    econ = backtest.economic_h1(pd.DataFrame(rows))
    assert list(econ["model"]) == ["ar5"]
    assert econ["turnover"].iloc[0] > 1  # flips every day
    assert econ["sharpe_net"].iloc[0] <= econ["sharpe_gross"].iloc[0] or econ["sharpe_gross"].iloc[0] == 0


def test_summaries_are_wellformed_on_noise():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=600)
    rows = []
    actual = rng.normal(0, 0.005, 600)
    for m in ("rw", "ar5"):
        fc = np.zeros(600) if m == "rw" else rng.normal(0, 0.0005, 600)
        rows.append(pd.DataFrame({"date": idx, "currency": "USD", "h": 1, "model": m, "forecast": fc, "actual": actual}))
    s = backtest.summarize_mean(pd.concat(rows))
    assert list(s["model"]) == ["ar5"]
    assert s["rel_rmse"].iloc[0] == pytest.approx(1.0, abs=0.02)
    assert s["dm_p_holm"].iloc[0] >= s["dm_p"].iloc[0]
    assert "ar5" in report.mean_table(s)
    assert "Holm" in report.mean_headline(s)


def test_splice_replaces_only_between_markers():
    text = "a\n<!-- x:start -->\nold\n<!-- x:end -->\nb"
    out = report.splice(text, "x", "new")
    assert out == "a\n<!-- x:start -->\nnew\n<!-- x:end -->\nb"
    with pytest.raises(ValueError):
        report.splice(text, "missing", "z")


def test_fetch_parses_the_ecb_csv(monkeypatch):
    csv = "CURRENCY,TIME_PERIOD,OBS_VALUE\nUSD,2024-01-02,1.10\nGBP,2024-01-02,0.86\nUSD,2024-01-03,1.09\nGBP,2024-01-03,0.87\n"

    class Resp:
        text = csv

        def raise_for_status(self):
            pass

    monkeypatch.setattr(data.httpx, "get", lambda *a, **k: Resp())
    frame = data.fetch(("USD", "GBP"))
    assert list(frame.columns) == ["USD", "GBP"]
    assert frame.loc["2024-01-03", "USD"] == pytest.approx(1.09)
    assert isinstance(io.StringIO(csv).read(), str)


def test_load_snapshot_rejects_non_positive_rates(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("date,USD\n2024-01-02,1.1\n2024-01-03,-1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        data.load_snapshot(p)
