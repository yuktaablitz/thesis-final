"""
Stage 4 — on-chain data. Tries CoinMetrics' free Community API first (no key
required, community-api.coinmetrics.io), which covers a solid chunk of
Tier 1/3 metrics: MVRV, realized cap/price, active addresses, tx count,
fees, hash rate. SOPR/aSOPR/NUPL and exchange in/out/net-flow (Tier 1/2)
are NOT available on CoinMetrics community — those need GLASSNODE_API_KEY
or CRYPTOQUANT_API_KEY in .env (paid tiers). This module degrades
gracefully: it always returns whatever it could fetch, and logs exactly
which configured metrics it could not get and why.

CoinMetrics docs: https://docs.coinmetrics.io/api/v4/
Community endpoint: https://community-api.coinmetrics.io/v4 (no key)
Paid endpoint:      https://api.coinmetrics.io/v4 (COINMETRICS_API_KEY)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.logging import get_logger

load_dotenv()
log = get_logger("onchain")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

COMMUNITY_BASE = "https://community-api.coinmetrics.io/v4"
PAID_BASE = "https://api.coinmetrics.io/v4"

# design-doc metric name -> CoinMetrics asset-metric name (community-available)
COINMETRICS_METRIC_MAP = {
    "mvrv": "CapMVRVCur",
    "realized_price": "CapRealUSD",   # divide by SplyCur downstream -> realized price
    "active_addresses": "AdrActCnt",
    "transaction_count": "TxCnt",
    "transaction_volume": "TxTfrValAdjUSD",
    "fees": "FeeTotUSD",
    "hash_rate": "HashRateMean",
    "supply": "SplyCur",
    "price_usd": "PriceUSD",
}
# Tier 1/2 metrics with no free community source — need Glassnode/CryptoQuant.
UNAVAILABLE_WITHOUT_PAID_KEY = [
    "mvrv_zscore", "sopr", "asopr", "nupl",
    "exchange_inflow", "exchange_outflow", "exchange_netflow", "exchange_reserve",
]


def _fetch_coinmetrics(asset: str, metrics: list[str], start: str, end: str,
                        api_key: str | None) -> pd.DataFrame:
    base = PAID_BASE if api_key else COMMUNITY_BASE
    url = f"{base}/timeseries/asset-metrics"
    params = {
        "assets": asset,
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "start_time": start,
        "end_time": end,
        "page_size": 10000,
    }
    if api_key:
        params["api_key"] = api_key

    rows = []
    while True:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            log.warning("CoinMetrics rate limited, sleeping 6s (community tier: 10 req/6s)")
            time.sleep(6)
            continue
        r.raise_for_status()
        payload = r.json()
        rows.extend(payload.get("data", []))
        next_token = payload.get("next_page_token")
        if not next_token:
            break
        params["next_page_token"] = next_token
        time.sleep(0.6)  # stay under community rate limit

    return pd.DataFrame(rows)


def fetch_onchain_features(asset: str = "btc", start: str = "2019-01-01",
                            end: str | None = None,
                            requested: list[str] | None = None) -> pd.DataFrame:
    """
    Returns a daily frame: [timestamp, <requested on-chain columns available>].
    `requested` defaults to everything in configs/approach1.yaml's onchain.tier1/2/3.
    Logs (does not raise) which requested metrics it could not source.
    """
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    api_key = os.environ.get("COINMETRICS_API_KEY") or None

    if requested is None:
        requested = list(COINMETRICS_METRIC_MAP.keys())

    coinmetrics_wanted = [r for r in requested if r in COINMETRICS_METRIC_MAP]
    cm_metric_names = [COINMETRICS_METRIC_MAP[r] for r in coinmetrics_wanted]

    df = pd.DataFrame()
    if cm_metric_names:
        log.info(f"fetching from CoinMetrics ({'paid' if api_key else 'community, no key'}): "
                  f"{cm_metric_names}")
        raw = _fetch_coinmetrics(asset, cm_metric_names, start, end, api_key)
        if not raw.empty:
            raw["timestamp"] = pd.to_datetime(raw["time"], utc=True).dt.normalize()
            rename = {v: k for k, v in COINMETRICS_METRIC_MAP.items()}
            for col in cm_metric_names:
                if col in raw.columns:
                    raw[col] = pd.to_numeric(raw[col], errors="coerce")
            df = raw.rename(columns=rename)[["timestamp", *coinmetrics_wanted]]

    if "realized_price" in df.columns and "supply" in df.columns:
        df["realized_price"] = df["realized_price"] / df["supply"]
    if "price_usd" in df.columns and "realized_price" in df.columns:
        df["price_to_realized_price"] = df["price_usd"] / df["realized_price"]

    missing = [r for r in requested if r in UNAVAILABLE_WITHOUT_PAID_KEY]
    have_paid_key = bool(os.environ.get("GLASSNODE_API_KEY") or os.environ.get("CRYPTOQUANT_API_KEY"))
    if missing and not have_paid_key:
        log.warning(
            f"NOT fetched (need GLASSNODE_API_KEY or CRYPTOQUANT_API_KEY in .env): {missing}. "
            "See src/data/onchain.py fetch_glassnode()/fetch_cryptoquant() TODOs."
        )
    elif missing and have_paid_key:
        log.warning(
            f"{missing} requested but no fetcher wired up yet — implement "
            "fetch_glassnode()/fetch_cryptoquant() below, matching the Glassnode/"
            "CryptoQuant docs for your subscribed endpoints, then merge into df here."
        )

    if not df.empty:
        path = RAW_DIR / "onchain_coinmetrics.parquet"
        df.to_parquet(path, index=False)
        log.info(f"saved {len(df)} daily on-chain rows -> {path}")
    return df


def fetch_glassnode(metrics: list[str], start: str, end: str) -> pd.DataFrame:
    """TODO: implement once GLASSNODE_API_KEY is set. https://docs.glassnode.com/"""
    raise NotImplementedError("Set GLASSNODE_API_KEY and implement against Glassnode's API docs")


def fetch_cryptoquant(metrics: list[str], start: str, end: str) -> pd.DataFrame:
    """TODO: implement once CRYPTOQUANT_API_KEY is set. https://docs.cryptoquant.com/"""
    raise NotImplementedError("Set CRYPTOQUANT_API_KEY and implement against CryptoQuant's API docs")


if __name__ == "__main__":
    df = fetch_onchain_features()
    print(df.tail())
    print(df.shape)
