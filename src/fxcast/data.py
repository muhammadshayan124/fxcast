"""ECB euro foreign-exchange reference rates (daily, units of foreign currency per 1 EUR).

Source: ECB Data Portal, dataset EXR (https://data-api.ecb.europa.eu). The rates are the ECB's
2.15 pm CET reference fixings. They are not revised after publication, so there is no
look-ahead from data vintages, which removes one common source of leakage in macro
forecasting. The snapshot used for every number in this repository is committed under `data/`
so results are exactly reproducible; `fxcast fetch` refreshes it.

Reuse of ECB statistics is permitted with attribution to the European Central Bank.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

CURRENCIES = ("USD", "GBP", "JPY", "CHF", "CNY", "AUD")
API = "https://data-api.ecb.europa.eu/service/data/EXR"
DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[2] / "data" / "ecb_exr.csv"


def fetch(currencies: tuple[str, ...] = CURRENCIES, start: str = "1999-01-04", timeout: float = 60.0) -> pd.DataFrame:
    """Download daily reference rates; returns a date-indexed frame with one column per currency."""
    url = f"{API}/D.{'+'.join(currencies)}.EUR.SP00.A"
    resp = httpx.get(url, params={"format": "csvdata", "startPeriod": start}, timeout=timeout)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), usecols=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"])
    wide = raw.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE")
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    return wide.sort_index()[list(currencies)]


def save_snapshot(frame: pd.DataFrame, path: Path = DEFAULT_SNAPSHOT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    meta = {
        "source": "ECB Data Portal, dataset EXR, D.<CCY>.EUR.SP00.A (units of currency per EUR)",
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": int(len(frame)), "first": str(frame.index[0].date()), "last": str(frame.index[-1].date()),
        "currencies": list(frame.columns),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_snapshot(path: Path = DEFAULT_SNAPSHOT) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="date", parse_dates=True)
    if (frame.dropna(how="all") <= 0).any().any():
        raise ValueError("exchange rates must be strictly positive")
    return frame


def log_levels(rates: pd.DataFrame) -> pd.DataFrame:
    return np.log(rates)


def daily_returns(rates: pd.DataFrame) -> pd.DataFrame:
    """Log returns r_t = log(S_t) - log(S_{t-1}). A return is only defined between two observed fixings."""
    return log_levels(rates).diff()
