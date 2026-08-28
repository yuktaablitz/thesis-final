"""
Binance BTCUSDT (or any symbol) spot klines. Public market-data endpoint —
no API key. Uses data-api.binance.vision (Binance's public read-only market
data mirror, same data as api.binance.com/api/v3) with a fallback to
api.binance.com if the mirror is unreachable.
"""
import time

import pandas as pd
import requests

from config import DATA_START, DATA_END, RAW_DIR, SYMBOL
from src.common import save_parquet

PRIMARY_URL = "https://data-api.binance.vision/api/v3/klines"
FALLBACK_URL = "https://api.binance.com/api/v3/klines"

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


def _get_klines(params: dict, retries: int = 5) -> list:
    last_exc = None
    for url in (PRIMARY_URL, FALLBACK_URL):
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                time.sleep(2 ** attempt)
        # primary exhausted its retries -> try the fallback host once
    raise RuntimeError(f"Failed to fetch klines from both hosts: {last_exc}")


def fetch_binance_spot(symbol: str = SYMBOL, interval: str = "1h",
                        start: str = DATA_START, end: str = DATA_END) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)

    cur = start_ms
    rows = []
    while cur < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cur,
                   "endTime": end_ms, "limit": 1000}
        batch = _get_klines(params)
        if not batch:
            break
        rows.extend(batch)

        next_cur = int(batch[-1][0]) + 1
        if next_cur <= cur:
            break
        cur = next_cur

        if len(rows) % 20000 < 1000:
            print(pd.to_datetime(cur, unit="ms", utc=True), len(rows))
        time.sleep(0.15)

    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df[KEEP_COLUMNS]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def run(force_refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / "binance_btcusdt_spot_1h.parquet"

    if path.exists() and not force_refresh:
        print("Loading cached Binance Spot")
        return pd.read_parquet(path)

    df = fetch_binance_spot()
    save_parquet(df, path)
    return df


if __name__ == "__main__":
    out = run()
    print(out.tail())
    print(out.shape)
