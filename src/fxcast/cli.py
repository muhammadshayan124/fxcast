from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

from fxcast import backtest, data, report

app = typer.Typer(help="Honest walk-forward FX forecasting backtests on ECB data.", no_args_is_help=True)
console = Console(highlight=False)
RESULTS = Path("results")


@app.command()
def fetch(
    out: Annotated[Path, typer.Option(help="Snapshot CSV to write")] = data.DEFAULT_SNAPSHOT,
) -> None:
    """Download ECB daily reference rates and save a dated snapshot."""
    frame = data.fetch()
    data.save_snapshot(frame, out)
    console.print(f"{len(frame)} rows, {frame.index[0].date()} to {frame.index[-1].date()} -> {out}")


@app.command("backtest")
def backtest_cmd(
    snapshot: Annotated[Path, typer.Option(help="Data snapshot")] = data.DEFAULT_SNAPSHOT,
    out: Annotated[Path, typer.Option(help="Results directory")] = RESULTS,
    currencies: Annotated[str, typer.Option(help="Comma-separated")] = ",".join(data.CURRENCIES),
    horizons: Annotated[str, typer.Option(help="Comma-separated forecast horizons in days")] = "1,5,20",
    jobs: Annotated[int, typer.Option(help="Parallel workers")] = 4,
    min_train: int = backtest.MIN_TRAIN,
    refit_every: int = backtest.REFIT_EVERY,
) -> None:
    """Run both tracks, write summaries, figures and run metadata."""
    out.mkdir(parents=True, exist_ok=True)
    rates = data.load_snapshot(snapshot)
    levels = data.log_levels(rates)
    curs = [c.strip() for c in currencies.split(",")]
    hs = tuple(int(x) for x in horizons.split(","))
    kw = {"min_train": min_train, "refit_every": refit_every}
    start = time.time()

    console.print("mean track ...")
    mean_preds = backtest.run_mean(levels, curs, hs, n_jobs=jobs, **kw)
    console.print("volatility track ...")
    vol_preds, fallbacks = backtest.run_vol(levels, curs, hs, n_jobs=jobs, **kw)

    mean_sum, vol_sum = backtest.summarize_mean(mean_preds), backtest.summarize_vol(vol_preds)
    econ = backtest.economic_h1(mean_preds)
    mean_sum.to_csv(out / "summary_mean.csv", index=False)
    vol_sum.to_csv(out / "summary_vol.csv", index=False)
    econ.to_csv(out / "economic_h1.csv", index=False)
    mean_preds.to_csv(out / "predictions_mean.csv.gz", index=False)
    vol_preds.to_csv(out / "predictions_vol.csv.gz", index=False)

    report.plot_mean(mean_sum, out / "mean_rel_rmse.png")
    report.plot_vol(vol_sum, out / "vol_qlike_gain.png")
    if "USD" in curs and 5 in hs:
        report.plot_cumulative_error(mean_preds, out / "cumulative_error_usd_h5.png", "USD", 5)

    meta = {
        "data_first": str(rates.index[0].date()), "data_last": str(rates.index[-1].date()), "rows": int(len(rates)),
        "currencies": curs, "horizons": list(hs), "min_train": min_train, "refit_every": refit_every,
        "first_forecast": str(mean_preds["date"].min()), "last_forecast": str(mean_preds["date"].max()),
        "n_forecasts_mean": int(len(mean_preds)), "garch_fallback_blocks": fallbacks,
        "seconds": round(time.time() - start), "python": platform.python_version(),
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    console.print(report.mean_headline(mean_sum))
    console.print(report.vol_headline(vol_sum))
    console.print(f"done in {meta['seconds']}s -> {out}")


@app.command("report")
def report_cmd(
    results: Annotated[Path, typer.Option(help="Results directory")] = RESULTS,
    readme: Annotated[Path, typer.Option(help="README with <!-- name:start/end --> markers to fill")] = Path("README.md"),
) -> None:
    """Fill the README's result sections from the summary CSVs, so the text can never drift from the data."""
    mean_sum = pd.read_csv(results / "summary_mean.csv")
    vol_sum = pd.read_csv(results / "summary_vol.csv")
    econ = pd.read_csv(results / "economic_h1.csv")
    meta = json.loads((results / "run.json").read_text(encoding="utf-8"))
    text = readme.read_text(encoding="utf-8")
    text = report.splice(text, "run", (
        f"Data: ECB daily reference rates, {meta['data_first']} to {meta['data_last']} ({meta['rows']:,} fixings). "
        f"Forecasts from {meta['first_forecast'][:10]} to {meta['last_forecast'][:10]} "
        f"({meta['n_forecasts_mean']:,} mean forecasts), refit every {meta['refit_every']} trading days after "
        f"{meta['min_train']} days of training. GARCH fit failures: {meta['garch_fallback_blocks']}."))
    text = report.splice(text, "mean-headline", report.mean_headline(mean_sum))
    text = report.splice(text, "mean-table", report.mean_table(mean_sum))
    text = report.splice(text, "vol-headline", report.vol_headline(vol_sum))
    text = report.splice(text, "vol-table", report.vol_table(vol_sum))
    text = report.splice(text, "econ-table", report.economic_table(econ))
    readme.write_text(text, encoding="utf-8")
    console.print(f"updated {readme}")


if __name__ == "__main__":
    app()
