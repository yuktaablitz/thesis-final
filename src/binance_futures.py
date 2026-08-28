"""
Binance USD-M perpetual futures — funding rate (REST, full history) and
daily metrics (open interest, top-trader ratios, taker ratio) from
Binance's public data archive (data.binance.vision), since the REST
`openInterestHist`/`*LongShortRatio` endpoints only retain ~30 days.

Binance USD-M futures launched 2019-09-13 — nothing exists before that;
callers should NOT drop earlier spot rows just because this returns empty
for Jan-Aug 2019 (see src/master_table.py's missingness handling).
"""
import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from config import DATA_END, DATA_START, FUTURES_LAUNCH, RAW_DIR, SYMBOL
from src.common import save_parquet

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
ARCHIVE_BASE = "https://data.binance.vision/data/futures/um/daily/metrics"


# ---------------------------------------------------------------- funding --
def fetch_funding(symbol: str = SYMBOL, start: str = DATA_START, end: str = DATA_END) -> pd.DataFrame:
    start_ts = max(pd.Timestamp(start), pd.Timestamp(FUTURES_LAUNCH))
    end_ts = pd.Timestamp(end)

    current = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    rows = []

    while current < end_ms:
        params = {"symbol": symbol, "startTime": current, "endTime": end_ms, "limit": 1000}
        for attempt in range(1, 6):
            try:
                r = requests.get(FUNDING_URL, params=params, timeout=30)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                print(f"funding fetch attempt {attempt}/5 failed: {e}")
                time.sleep(2 ** attempt)
        else:
            break

        if not batch:
            break
        rows.extend(batch)

        nxt = int(batch[-1]["fundingTime"]) + 1
        if nxt <= current:
            break
        current = nxt
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return (
        df[["timestamp", "funding_rate"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def run_funding(force_refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / "binance_btcusdt_funding.parquet"
    if path.exists() and not force_refresh:
        print("Loading cached funding rate")
        return pd.read_parquet(path)
    df = fetch_funding()
    save_parquet(df, path)
    return df


# --------------------------------------------------- archived daily metrics --
def read_binance_zip(url: str) -> pd.DataFrame | None:
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_name = z.namelist()[0]
    with z.open(csv_name) as f:
        return pd.read_csv(f)


def fetch_futures_daily_metrics(symbol: str = SYMBOL, start: str = DATA_START,
                                 end: str = DATA_END) -> pd.DataFrame:
    dates = pd.date_range(
        start=max(pd.Timestamp(start), pd.Timestamp(FUTURES_LAUNCH)),
        end=pd.Timestamp(end), freq="D", inclusive="left",
    )

    frames = []
    for date in dates:
        ds = date.strftime("%Y-%m-%d")
        url = f"{ARCHIVE_BASE}/{symbol}/{symbol}-metrics-{ds}.zip"
        try:
            df = read_binance_zip(url)
            if df is not None:
                frames.append(df)
        except Exception as e:
            print("metrics failed:", ds, e)
        time.sleep(0.05)

    if not frames:
        print("No archived futures metrics found — this is expected for dates "
              "before the archive's retention window; not fatal.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def normalize_futures_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    df = raw.copy()

    if "create_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["create_time"], utc=True, errors="coerce")
    else:
        raise ValueError("expected a 'create_time' column in Binance's archived metrics CSV")

    numeric_cols = [
        "sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rename = {
        "sum_open_interest": "futures_open_interest",
        "sum_open_interest_value": "futures_open_interest_usd",
        "count_toptrader_long_short_ratio": "top_trader_account_ratio",
        "sum_toptrader_long_short_ratio": "top_trader_position_ratio",
        "count_long_short_ratio": "global_long_short_ratio",
        "sum_taker_long_short_vol_ratio": "futures_taker_ratio",
    }
    df = df.rename(columns=rename)

    keep = ["timestamp"] + [v for v in rename.values() if v in df.columns]
    return (
        df[keep].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    )


def run_futures_metrics(force_refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / "binance_btcusdt_futures_metrics.parquet"
    if path.exists() and not force_refresh:
        print("Loading cached futures metrics")
        return pd.read_parquet(path)
    raw = fetch_futures_daily_metrics()
    df = normalize_futures_metrics(raw)
    if not df.empty:
        save_parquet(df, path)
    return df


# --------------------------------------------------------- derived features --
def add_derivatives_features(df: pd.DataFrame) -> pd.DataFrame:
    """Call AFTER merging funding/futures metrics onto the hourly master clock (src/master_table.py)."""
    df = df.sort_values("timestamp").copy()

    if "futures_open_interest" in df.columns:
        for h in [1, 6, 24]:
            df[f"oi_change_{h}"] = df["futures_open_interest"].pct_change(h)
        mean = df["futures_open_interest"].rolling(168).mean()
        std = df["futures_open_interest"].rolling(168).std()
        df["oi_zscore_7d"] = (df["futures_open_interest"] - mean) / std

    if "funding_rate" in df.columns:
        df["funding_mean_24h"] = df["funding_rate"].rolling(24).mean()
        df["funding_mean_72h"] = df["funding_rate"].rolling(72).mean()
        roll = df["funding_rate"].rolling(720)
        df["funding_zscore"] = (df["funding_rate"] - roll.mean()) / roll.std()

    if {"ret_1h", "oi_change_1h"} <= set(df.columns):
        df["ret1h_x_oichange1h"] = df["ret_1h"] * df["oi_change_1h"]
    if {"ret_24h", "oi_change_24h"} <= set(df.columns):
        df["ret24h_x_oichange24h"] = df["ret_24h"] * df["oi_change_24h"]
    if {"funding_rate", "oi_change_24h"} <= set(df.columns):
        df["funding_x_oichange24h"] = df["funding_rate"] * df["oi_change_24h"]
    if {"funding_rate", "ret_24h"} <= set(df.columns):
        df["funding_x_ret24h"] = df["funding_rate"] * df["ret_24h"]

    return df.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    funding = run_funding()
    metrics = run_futures_metrics()
    print("funding:", funding.shape)
    print("futures metrics:", metrics.shape)
