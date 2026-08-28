"""
Small macro set (Stage 8), sourced via yfinance — no API key required.
Tickers come from configs/approach1.yaml `macro_tickers` so they stay in
one place. Note: yfinance's ^IRX/^TNX are yield-index quotes (already in
percent), not FRED's DGS2/DGS10 series — close enough for this feature set;
swap in FRED (needs a free key) later if you want the exact Treasury series.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger("macro")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_macro(start: str = "2019-01-01", end: str | None = None) -> pd.DataFrame:
    """Daily macro frame, one column per configured ticker, plus each column's 1-day return."""
    cfg = load_config("approach1.yaml")
    tickers: dict[str, str] = cfg["macro_tickers"]
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    frames = []
    for name, ticker in tickers.items():
        log.info(f"downloading {name} ({ticker})")
        hist = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if hist.empty:
            log.warning(f"no data for {ticker} — skipping")
            continue
        s = hist["Close"].rename(name)
        frames.append(s)

    if not frames:
        raise RuntimeError("No macro tickers returned data")

    df = pd.concat(frames, axis=1).reset_index().rename(columns={"Date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    for name in tickers:
        if name in df.columns:
            df[f"{name}_return_1d"] = df[name].pct_change()

    df = df.sort_values("timestamp").reset_index(drop=True)
    path = RAW_DIR / "macro.parquet"
    df.to_parquet(path, index=False)
    log.info(f"saved {len(df)} daily macro rows -> {path}")
    return df


def load_or_download(start: str = "2019-01-01", end: str | None = None) -> pd.DataFrame:
    return fetch_macro(start=start, end=end)


if __name__ == "__main__":
    df = fetch_macro()
    print(df.tail())
