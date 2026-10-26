"""Why CHF is the one currency where EWMA/GARCH lose on variance: the SNB removed the EUR/CHF floor
on 15 Jan 2015. Reports the worst days and the result with Jan-Feb 2015 excluded.
Run after `fxcast backtest`."""

import pandas as pd

from fxcast import metrics

p = pd.read_csv("results/predictions_vol.csv.gz", parse_dates=["date"])
g = p[(p.currency == "CHF") & (p.h == 5)]
w = g.pivot(index="date", columns="model", values="forecast")
a = g.groupby("date")["actual"].first()
keep = ~((w.index >= "2015-01-01") & (w.index <= "2015-02-28"))
for m in ("ewma", "garch", "har"):
    d = pd.Series(metrics.qlike(w[m].to_numpy(), a.to_numpy()) - metrics.qlike(w["expanding"].to_numpy(), a.to_numpy()), index=w.index)
    r = metrics.dm_test(
        metrics.qlike(w[m].to_numpy()[keep], a.to_numpy()[keep]),
        metrics.qlike(w["expanding"].to_numpy()[keep], a.to_numpy()[keep]),
        5,
    )
    print(f"{m}: worst day {d.idxmax().date()} (excess loss {d.max():.0f}); excluding Jan-Feb 2015: QLIKE gain {-r.mean_diff:+.3f}, p={r.p_value:.4f}")
