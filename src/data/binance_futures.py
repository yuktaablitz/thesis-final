"""
Binance USDⓈ-M perpetual futures: funding rate, premium index (for basis),
and open interest / long-short ratios. Public REST — no API key required.

IMPORTANT — history retention limit:
Binance's public API only retains ~30 days of history for
`openInterestHist`, `globalLongShortAccountRatio`, and
`topLongShort*Ratio` (Stage 3 "long/short positioning" + "open interest").
Funding rate and premium index klines DO have full history back to
contract listing. So out of the box you get, for the full 2019-2025 range:
  - funding_rate, funding_mean_*, funding_zscore        -> full history
  - premium_index / perp_spot_basis / basis_change       -> full history
  - open_interest, oi_change_*, long/short ratios         -> last ~30 days only

For full-history OI / positioning you need a paid vendor (Coinglass API,
Amberdata, Kaiko, etc.) — see the TODO in fetch_open_interest_hist below.
Everything here is still worth collecting now: it's free, it back-fills as
time passes, and funding+basis alone cover a meaningful chunk of Stage 3.

Docs:
  https://binance-docs.github.io/apidocs/futures/en/#funding-rate-history
  https://binance-docs.github.io/apidocs/futures/en/#premium-index-kline-data
  https://binance-docs.github.io/apidocs/futures/en/#open-interest-statistics
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from src.utils.logging import get_logger

log = get_logger("binance_futures")

BASE = "https://fapi.binance.com"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "binance"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _get(path: str, params: dict, retries: int = 5) -> list:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"{path} attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {path} after {retries} retries")


def _paginate(path: str, symbol: str, start_ms: int, end_ms: int, extra: dict,
              time_key_out: str, limit: int = 1000, step_field: str = "fundingTime") -> pd.DataFrame:
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": limit, **extra}
        page = _get(path, params)
        if not page:
            break
        rows.extend(page)
        last_ts = page[-1].get(step_field)
        if last_ts is None or last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    if not df.empty and step_field in df.columns:
        df = df.rename(columns={step_field: time_key_out})
    return df


def fetch_funding_rate(symbol: str = "BTCUSDT", start: str = "2019-01-01",
                        end: str | None = None) -> pd.DataFrame:
    """Full-history funding rate (8h cadence on Binance)."""
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    df = _paginate("/fapi/v1/fundingRate", symbol, start_ms, end_ms, {}, "timestamp",
                    step_field="fundingTime")
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"])
    df = df[["timestamp", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
    path = RAW_DIR / f"{symbol}_funding_rate.parquet"
    df.to_parquet(path, index=False)
    log.info(f"saved {len(df)} funding rows -> {path}")
    return df


def fetch_premium_index_klines(symbol: str = "BTCUSDT", interval: str = "1h",
                                start: str = "2019-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Premium index klines: OHLC of (mark_price - index_price) / index_price.
    Used to derive perp_spot_basis / basis_change / basis_zscore without
    needing a separate spot-vs-mark alignment step.
    """
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cursor,
                   "endTime": end_ms, "limit": 1500}
        page = _get("/fapi/v1/premiumIndexKlines", params)
        if not page:
            break
        rows.extend(page)
        cursor = page[-1][6] + 1
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame()

    cols = ["open_time", "open", "high", "low", "close", "ignore1", "close_time",
            "ignore2", "ignore3", "ignore4", "ignore5", "ignore6"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c])
    df = df.rename(columns={"close": "premium_index"})[["timestamp", "premium_index"]]
    path = RAW_DIR / f"{symbol}_premium_index_{interval}.parquet"
    df.to_parquet(path, index=False)
    log.info(f"saved {len(df)} premium-index rows -> {path}")
    return df


def fetch_open_interest_hist(symbol: str = "BTCUSDT", period: str = "1h",
                              start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Only returns the last ~30 days — Binance does not retain more via the
    public API. TODO: once you have a paid vendor key (Coinglass/Amberdata/
    Kaiko), replace this function's body with that vendor's call and keep
    the same output contract: columns [timestamp, open_interest].
    """
    end_ms = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()).timestamp() * 1000)
    start_ms = int((pd.Timestamp(start, tz="UTC") if start
                    else pd.Timestamp.utcnow() - pd.Timedelta(days=30)).timestamp() * 1000)
    df = _paginate("/futures/data/openInterestHist", symbol, start_ms, end_ms,
                    {"period": period}, "timestamp", limit=500, step_field="timestamp")
    if df.empty:
        log.warning("openInterestHist returned nothing — check the ~30 day retention window")
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"])
    df = df[["timestamp", "open_interest"]]
    path = RAW_DIR / f"{symbol}_open_interest_{period}.parquet"
    df.to_parquet(path, index=False)
    log.info(f"saved {len(df)} OI rows -> {path} (last ~30d only, see docstring)")
    return df


def fetch_long_short_ratio(symbol: str = "BTCUSDT", period: str = "1h",
                            kind: str = "global") -> pd.DataFrame:
    """
    kind: "global" (globalLongShortAccountRatio), "top_account"
    (topLongShortAccountRatio), or "top_position" (topLongShortPositionRatio).
    Same ~30 day retention limit as open interest — see fetch_open_interest_hist.
    """
    endpoint = {
        "global": "/futures/data/globalLongShortAccountRatio",
        "top_account": "/futures/data/topLongShortAccountRatio",
        "top_position": "/futures/data/topLongShortPositionRatio",
    }[kind]
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = end_ms - int(pd.Timedelta(days=30).total_seconds() * 1000)
    df = _paginate(endpoint, symbol, start_ms, end_ms, {"period": period}, "timestamp",
                    limit=500, step_field="timestamp")
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["long_short_ratio"] = pd.to_numeric(df["longShortRatio"])
    return df[["timestamp", "long_short_ratio"]].rename(
        columns={"long_short_ratio": f"{kind}_long_short_ratio"}
    )


if __name__ == "__main__":
    funding = fetch_funding_rate()
    premium = fetch_premium_index_klines()
    oi = fetch_open_interest_hist()
    print("funding:", funding.shape, "premium:", premium.shape, "oi (last 30d):", oi.shape)
