"""
Alternative.me Crypto Fear & Greed Index. No API key. Daily since Feb 2018.

Point-in-time rule: a day's F&G value is only fully "known" once that day
has completed, so `available_at = source_time + 1 day` — no same-day
leakage, no bfill.
"""
import pandas as pd
import requests

from config import RAW_DIR
from src.common import save_parquet


def fetch_fng() -> pd.DataFrame:
    url = "https://api.alternative.me/fng/"
    params = {"limit": 0, "format": "json"}  # limit=0 -> full history

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]

    df = pd.DataFrame(data)
    df["source_time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.normalize()
    df["fear_greed"] = pd.to_numeric(df["value"], errors="coerce")

    df = df[["source_time", "fear_greed"]].sort_values("source_time").drop_duplicates("source_time")
    df["available_at"] = df["source_time"] + pd.Timedelta(days=1)

    return df.reset_index(drop=True)


def run_fng(force_refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / "fear_greed_index.parquet"
    if path.exists() and not force_refresh:
        print("Loading cached Fear & Greed")
        return pd.read_parquet(path)
    df = fetch_fng()
    save_parquet(df, path)
    return df


if __name__ == "__main__":
    df = run_fng()
    print(df.tail(10).to_string(index=False))
