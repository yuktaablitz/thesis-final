"""
Binance BTCUSDT (or any symbol) spot klines, 1h base interval. Public REST —
no API key required. Paginates in 1000-candle chunks, checkpoints to
data/raw/binance/, and resumes automatically if interrupted.

Docs: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data

Usage:
    python -m src.data.binance_spot --symbol BTCUSDT --start 2019-01-01 --end 2026-08-28
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from src.utils.logging import get_logger

log = get_logger("binance_spot")

BASE_URL = "https://api.binance.com/api/v3/klines"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "binance"
RAW_DIR.mkdir(parents=True, exist_ok=True)

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
KEEP_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
]
NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
]


def _checkpoint_path(symbol: str, interval: str) -> Path:
    return RAW_DIR / f"{symbol}_{interval}_spot.parquet"


def _fetch_page(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000,
                 retries: int = 5) -> list:
    params = {
        "symbol": symbol, "interval": interval,
        "startTime": start_ms, "endTime": end_ms, "limit": limit,
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            if r.status_code == 429:
                wait = 2 ** attempt
                log.warning(f"rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch klines after {retries} retries")


def download_spot_klines(symbol: str = "BTCUSDT", interval: str = "1h",
                          start: str = "2019-01-01", end: str | None = None,
                          resume: bool = True) -> pd.DataFrame:
    """Download (or resume) full spot kline history and return the merged, deduped frame."""
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    path = _checkpoint_path(symbol, interval)
    existing = pd.read_parquet(path) if (resume and path.exists()) else pd.DataFrame()
    if not existing.empty:
        last_ts = int(existing["open_time"].max())
        start_ms = max(start_ms, last_ts + 1)
        log.info(f"resuming {symbol} {interval} from {pd.to_datetime(start_ms, unit='ms')}")

    rows = []
    cursor = start_ms
    interval_ms = 3_600_000 if interval == "1h" else None
    if interval_ms is None:
        raise ValueError("this helper is tuned for interval='1h'; extend it for other intervals")

    while cursor < end_ms:
        page = _fetch_page(symbol, interval, cursor, end_ms)
        if not page:
            break
        rows.extend(page)
        cursor = page[-1][6] + 1  # close_time of last candle + 1ms
        if len(rows) % 20000 < 1000:
            log.info(f"  fetched {len(rows)} candles so far, at {pd.to_datetime(cursor, unit='ms')}")
        time.sleep(0.15)  # be polite to the public rate limit

    if rows:
        new_df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    else:
        combined = existing

    combined = combined.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    combined.to_parquet(path, index=False)
    log.info(f"saved {len(combined)} candles -> {path}")
    return _postprocess(combined)


def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c])
    return df[KEEP_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def load_or_download(symbol: str = "BTCUSDT", interval: str = "1h",
                      start: str = "2019-01-01", end: str | None = None) -> pd.DataFrame:
    """Convenience entry point for other modules: resumes/updates then returns clean OHLCV."""
    return download_spot_klines(symbol=symbol, interval=interval, start=start, end=end, resume=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    out = download_spot_klines(args.symbol, args.interval, args.start, args.end)
    print(out.tail())
    print(out.shape)
