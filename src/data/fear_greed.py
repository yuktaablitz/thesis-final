"""
Alternative.me Crypto Fear & Greed Index. No API key required. Daily
granularity since Feb 2018 — merge point-in-time only (no bfill; see
src/features/technical.py merge helpers and Stage 5 in the design doc).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.utils.logging import get_logger

log = get_logger("fear_greed")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_fear_greed_index() -> pd.DataFrame:
    url = "https://api.alternative.me/fng/"
    params = {"limit": 0, "format": "json"}  # limit=0 -> full history

    log.info("fetching Alternative.me Fear & Greed Index (all history)")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise ValueError("Empty response from alternative.me FNG API")

    df = pd.DataFrame(
        {
            "timestamp": [pd.to_datetime(int(d["timestamp"]), unit="s", utc=True) for d in data],
            "fear_greed": [int(d["value"]) for d in data],
            "fng_label": [d["value_classification"] for d in data],
        }
    )
    df["timestamp"] = df["timestamp"].dt.normalize()
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    path = RAW_DIR / "fear_greed_index.parquet"
    df.to_parquet(path, index=False)
    log.info(f"saved {len(df)} daily FNG rows -> {path}")
    return df


def load_or_download() -> pd.DataFrame:
    path = RAW_DIR / "fear_greed_index.parquet"
    if path.exists():
        # Cheap daily series — just refresh every call rather than tracking staleness.
        try:
            return fetch_fear_greed_index()
        except Exception as e:
            log.warning(f"refresh failed ({e}), using cached copy")
            return pd.read_parquet(path)
    return fetch_fear_greed_index()


if __name__ == "__main__":
    df = load_or_download()
    print(df.tail(10).to_string(index=False))
