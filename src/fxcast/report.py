"""Tables (Markdown) and figures for the backtest results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ALPHA = 0.05


def _star(p: float) -> str:
    return "*" if p < ALPHA else ""


def mean_table(summary: pd.DataFrame) -> str:
    """Relative RMSE vs the random walk (below 1.000 = better), starred if the Holm-adjusted DM p < 0.05."""
    lines = []
    for h, g in summary.groupby("h"):
        models = list(g["model"].unique())
        lines += [f"**h = {h} day{'s' if h > 1 else ''}**", "", "| currency | " + " | ".join(models) + " |",
                  "|---|" + "---|" * len(models)]
        for cur, gc in g.groupby("currency"):
            cells = []
            for m in models:
                row = gc[gc["model"] == m].iloc[0]
                cells.append(f"{row.rel_rmse:.4f}{_star(row.dm_p_holm)}")
            lines.append(f"| {cur} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def mean_headline(summary: pd.DataFrame) -> str:
    n = len(summary)
    sig = summary[summary["dm_p_holm"] < ALPHA]
    better = sig[sig["rel_rmse"] < 1]
    worse = sig[sig["rel_rmse"] > 1]
    raw = int((summary["dm_p"] < ALPHA).sum())
    beat = int((summary["rel_rmse"] < 1).sum())
    lines = [
        f"- {n} model-currency-horizon comparisons against the random walk.",
        f"- {beat} of {n} have a lower RMSE than the random walk; {raw} of {n} are 'significant' at 5% **before** multiplicity correction.",
        f"- After Holm correction: **{len(better)}** significantly better than the random walk, **{len(worse)}** significantly worse.",
    ]
    if len(better):
        lines.append("- Significantly better: " + "; ".join(
            f"{r.model}/{r.currency}/h={r.h} (rel. RMSE {r.rel_rmse:.4f}, adj. p={r.dm_p_holm:.3f})" for r in better.itertuples()))
    best = summary.sort_values("rel_rmse").iloc[0]
    lines.append(f"- Best single result: {best.model} on {best.currency} at h={best.h}: relative RMSE {best.rel_rmse:.4f} "
                 f"(adj. p={best.dm_p_holm:.2f}), i.e. {100 * (1 - best.rel_rmse):.2f}% lower RMSE.")
    return "\n".join(lines)


def vol_table(summary: pd.DataFrame) -> str:
    """QLIKE gain over the expanding-variance benchmark (positive = better), starred if Holm-adjusted p < 0.05."""
    lines = []
    for h, g in summary.groupby("h"):
        models = list(g["model"].unique())
        lines += [f"**h = {h} day{'s' if h > 1 else ''}**", "", "| currency | " + " | ".join(models) + " |",
                  "|---|" + "---|" * len(models)]
        for cur, gc in g.groupby("currency"):
            cells = []
            for m in models:
                row = gc[gc["model"] == m].iloc[0]
                cells.append(f"{row.qlike_gain:+.4f}{_star(row.dm_p_holm)}")
            lines.append(f"| {cur} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def vol_headline(summary: pd.DataFrame) -> str:
    n = len(summary)
    sig = summary[(summary["dm_p_holm"] < ALPHA) & (summary["qlike_gain"] > 0)]
    worse = summary[(summary["dm_p_holm"] < ALPHA) & (summary["qlike_gain"] < 0)]
    by_model = summary.assign(win=(summary["dm_p_holm"] < ALPHA) & (summary["qlike_gain"] > 0)).groupby("model")["win"].agg(["sum", "count"])
    parts = ", ".join(f"{m}: {int(r['sum'])}/{int(r['count'])}" for m, r in by_model.iterrows())
    return (f"- {n} comparisons against the expanding-variance benchmark. After Holm correction, **{len(sig)}** are significantly better "
            f"and {len(worse)} significantly worse.\n- Significantly better, by model: {parts}.")


def economic_table(econ: pd.DataFrame) -> str:
    wide = econ.pivot(index="currency", columns="model", values="sharpe_net")
    lines = ["| currency | " + " | ".join(wide.columns) + " |", "|---|" + "---|" * len(wide.columns)]
    for cur, row in wide.iterrows():
        lines.append(f"| {cur} | " + " | ".join(f"{v:+.2f}" for v in row) + " |")
    return "\n".join(lines)


# -- figures -----------------------------------------------------------------------------------

def plot_mean(summary: pd.DataFrame, path: Path) -> None:
    horizons = sorted(summary["h"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 3.8), sharey=True)
    for ax, h in zip(np.atleast_1d(axes), horizons):
        g = summary[summary["h"] == h]
        p = g.pivot(index="currency", columns="model", values="rel_rmse")
        p.plot.bar(ax=ax, width=0.8, legend=ax is np.atleast_1d(axes)[-1])
        ax.axhline(1.0, color="black", lw=1)
        ax.set_title(f"h = {h}")
        ax.set_xlabel("")
        ax.set_ylim(min(0.97, p.min().min() - 0.01), max(1.04, p.max().max() + 0.01))
    np.atleast_1d(axes)[0].set_ylabel("RMSE relative to random walk (<1 is better)")
    fig.suptitle("Return forecasts: no model reliably beats the random walk")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_vol(summary: pd.DataFrame, path: Path) -> None:
    horizons = sorted(summary["h"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 3.8), sharey=True)
    for ax, h in zip(np.atleast_1d(axes), horizons):
        g = summary[summary["h"] == h]
        g.pivot(index="currency", columns="model", values="qlike_gain").plot.bar(ax=ax, width=0.8, legend=ax is np.atleast_1d(axes)[-1])
        ax.axhline(0.0, color="black", lw=1)
        ax.set_title(f"h = {h}")
        ax.set_xlabel("")
    np.atleast_1d(axes)[0].set_ylabel("QLIKE gain over constant variance (>0 is better)")
    fig.suptitle("Variance forecasts: clustering is predictable")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cumulative_error(preds: pd.DataFrame, path: Path, currency: str = "USD", h: int = 5, models=("ridge", "hgb")) -> None:
    """Cumulative squared-error difference vs the random walk. A model that beats it has a line
    that trends up; a flat or wandering line is noise. Looking at it over time shows whether any
    apparent edge is steady or comes from one episode."""
    g = preds[(preds["currency"] == currency) & (preds["h"] == h)]
    wide = g.pivot(index="date", columns="model", values="forecast")
    actual = g.groupby("date")["actual"].first().loc[wide.index]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for m in models:
        diff = (actual - wide["rw"]) ** 2 - (actual - wide[m]) ** 2
        (diff.cumsum() * 1e4).plot(ax=ax, label=m)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("cumulative squared-error gain vs RW (x1e4)")
    ax.set_title(f"{currency}/EUR, h = {h}: cumulative accuracy gain over the random walk")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def splice(readme: str, marker: str, content: str) -> str:
    """Replace everything between <!-- marker:start --> and <!-- marker:end -->."""
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    i, j = readme.find(start), readme.find(end)
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"README is missing the {marker} markers")
    return readme[: i + len(start)] + "\n" + content.strip("\n") + "\n" + readme[j:]
