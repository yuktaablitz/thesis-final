"""
Small, fixed macro set via FRED's public CSV endpoint — no API key needed.
Binance-only rule applies to crypto market/derivatives data, not macro;
FRED is used here instead of yfinance because it's the authoritative source
for these series and needs no key. Series: config.FRED_SERIES (Fed Funds,
2y/10y Treasury, VIX, broad USD index) — deliberately small, not 40 variables.

Point-in-time rule: a daily FRED observation is only usable starting the
NEXT day (`available_at = source_time + 1 day`), same convention as
Fear & Greed and on-chain.
"""
from io import StringIO

import pandas as pd
import requests

from config import DATA_END, DATA_START, FRED_SERIES, RAW_DIR
from src.common import save_parquet

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series: str) -> pd.DataFrame:
    r = requests.get(FRED_CSV_URL, params={"id": series}, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df.columns = ["date", series]
    df["source_time"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df[series] = pd.to_numeric(df[series], errors="coerce")
    df["available_at"] = df["source_time"] + pd.Timedelta(days=1)

    return df[["source_time", "available_at", series]].dropna(subset=["source_time"])


def run_macro(force_refresh: bool = False, start: str = DATA_START, end: str = DATA_END) -> list[pd.DataFrame]:
    """Returns one frame per FRED series (kept separate — master_table merges each with its own
    per-series available_at column so point-in-time audits stay per-source)."""
    frames = []
    for series in FRED_SERIES:
        path = RAW_DIR / f"fred_{series.lower()}.parquet"
        if path.exists() and not force_refresh:
            print(f"Loading cached {series}")
            df = pd.read_parquet(path)
        else:
            print(f"fetching FRED {series}")
            df = fetch_fred_series(series)
            if not df.empty:
                save_parquet(df, path)

        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        df = df[(df["source_time"] >= start_ts) & (df["source_time"] < end_ts)]
        frames.append(df.reset_index(drop=True))
    return frames


if __name__ == "__main__":
    for series, df in zip(FRED_SERIES, run_macro()):
        print(series, df.shape, df.tail(2).to_string(index=False))
