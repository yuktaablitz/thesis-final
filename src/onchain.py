"""
On-chain data via CoinMetrics' free Community API (no key). Discovery-first:
query the catalog for what BTC metrics are actually free before assuming any
specific metric (SOPR/NUPL/exchange-flows) is available — Semester 1's plan
assumed a fixed metric list; several of those are Glassnode/CryptoQuant-paid
only, not on CoinMetrics community.

Point-in-time rule: a daily on-chain metric is only usable starting the
NEXT day (`available_at = source_time + 1 day`), matching Fear & Greed.
"""
import pandas as pd
import requests

from config import CACHE_DIR, DATA_END, DATA_START, ONCHAIN_KEYWORDS, RAW_DIR
from src.common import save_parquet

BASE = "https://community-api.coinmetrics.io/v4"


def coinmetrics_catalog(asset: str = "btc") -> list[dict]:
    url = f"{BASE}/catalog-all-v2/asset-metrics"
    r = requests.get(url, params={"assets": asset, "page_size": 10000}, timeout=60)
    r.raise_for_status()
    return r.json()["data"]


def discover_metrics(asset: str = "btc", keywords: list[str] = ONCHAIN_KEYWORDS) -> list[dict]:
    """Returns the catalog entries whose metric name matches any keyword. Run this BEFORE
    fetch_coinmetrics — don't assume a metric (e.g. SOPR) is actually free until you see it here."""
    catalog = coinmetrics_catalog(asset)
    btc = next((row for row in catalog if row["asset"] == asset), None)
    if btc is None:
        return []
    return [m for m in btc["metrics"] if any(k in m["metric"].lower() for k in keywords)]


def fetch_coinmetrics(metrics: list[str], asset: str = "btc",
                       start: str = DATA_START, end: str = DATA_END) -> pd.DataFrame:
    url = f"{BASE}/timeseries/asset-metrics"
    params = {
        "assets": asset, "metrics": ",".join(metrics), "frequency": "1d",
        "start_time": start, "end_time": end, "page_size": 10000,
    }

    rows = []
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        rows.extend(payload.get("data", []))

        next_url = payload.get("next_page_url")
        next_token = payload.get("next_page_token")
        if next_url:
            url, params = next_url, None
        elif next_token:
            params["next_page_token"] = next_token
        else:
            url = None

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["source_time"] = pd.to_datetime(df["time"], utc=True)
    for col in metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["available_at"] = df["source_time"] + pd.Timedelta(days=1)
    keep = ["source_time", "available_at"] + [c for c in metrics if c in df.columns]
    return df[keep].drop_duplicates("source_time").sort_values("source_time").reset_index(drop=True)


def run_onchain(force_refresh: bool = False) -> pd.DataFrame:
    """Discovers available metrics, fetches them, caches, and prints what was requested-but-unavailable."""
    path = RAW_DIR / "onchain_coinmetrics.parquet"
    if path.exists() and not force_refresh:
        print("Loading cached on-chain data")
        return pd.read_parquet(path)

    discovered = discover_metrics()
    metric_names = sorted({m["metric"] for m in discovered})
    print(f"CoinMetrics community catalog matched {len(metric_names)} metrics: {metric_names}")

    if not metric_names:
        print("No matching free CoinMetrics metrics found — skipping on-chain")
        return pd.DataFrame()

    df = fetch_coinmetrics(metric_names)
    if not df.empty:
        save_parquet(df, path)
        cache_note = CACHE_DIR / "onchain_discovered_metrics.txt"
        cache_note.write_text("\n".join(metric_names))
    return df


if __name__ == "__main__":
    df = run_onchain()
    print(df.tail())
    print(df.shape)
