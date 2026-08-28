"""Shared helpers used by every source module — everything goes UTC."""
from pathlib import Path

import pandas as pd


def ensure_utc(series: pd.Series) -> pd.Series:
    """Convert a timestamp-like Series to UTC, coercing unparseable values to NaT."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def save_parquet(df: pd.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def hourly_clock(start, end) -> pd.DataFrame:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return pd.DataFrame({
        "timestamp": pd.date_range(start=start, end=end, freq="1h", inclusive="left", tz="UTC")
    })


def coverage_report(name: str, df: pd.DataFrame, timestamp_col: str = "timestamp") -> None:
    if df.empty:
        print(f"{name:25s} EMPTY")
        return
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    print(f"{name:25s}", f"rows={len(df):,}", f"| {ts.min()} -> {ts.max()}")


def get_or_fetch(path, fetch_fn, force_refresh: bool = False, **fetch_kwargs) -> pd.DataFrame:
    """
    Generic cache wrapper: if `path` exists and force_refresh is False, load
    it; otherwise call fetch_fn(**fetch_kwargs) -> DataFrame, save, return it.
    Every downloader in run_pipeline.py goes through this — never recompute
    an expensive fetch (Binance backfill, FinBERT, LLM classification) that's
    already on disk.
    """
    path = Path(path)
    if path.exists() and not force_refresh:
        print(f"[cache] {path.name}")
        return load_parquet(path)
    print(f"[fetch] {path.name}")
    df = fetch_fn(**fetch_kwargs)
    if df is not None and not df.empty:
        save_parquet(df, path)
    return df if df is not None else pd.DataFrame()
