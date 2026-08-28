"""
BTC news via CryptoCompare. Requires CC_API_KEY in .env — no default, no
hardcoded date range (that was the Semester 1 bug: START_DATE/END_DATE were
hardcoded in the collector cell, restricting news to 2025-2026 even though
the thesis window is 2019-2026). Dates come from config.DATA_START/DATA_END
like every other module.

Resumable: checkpoints to data/raw/cryptocompare_news.parquet every 20
pages, walking backward in time via `lTs`, so a multi-year backfill
survives interruption.
"""
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from config import DATA_END, DATA_START, RAW_DIR
from src.common import save_parquet

load_dotenv()

NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
CHECKPOINT_PATH = RAW_DIR / "cryptocompare_news.parquet"


def _api_key() -> str:
    key = os.environ.get("CC_API_KEY")
    if not key:
        raise RuntimeError("CC_API_KEY not set — add it to .env "
                            "(https://www.cryptocompare.com/cryptopian/api-keys)")
    return key


def _fetch_page(before_ts: int, api_key: str, retries: int = 5) -> list[dict]:
    params = {"lang": "EN", "categories": "BTC", "lTs": before_ts, "api_key": api_key}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(NEWS_URL, params=params, timeout=45)
            r.raise_for_status()
            return r.json().get("Data", [])
        except Exception as e:
            print(f"news fetch attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 ** attempt)
    return []


def _checkpoint(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df = df.drop_duplicates(subset="url").sort_values("published_at").reset_index(drop=True)
    save_parquet(df, CHECKPOINT_PATH)
    return df


def collect_btc_news(start: str = DATA_START, end: str = DATA_END, resume: bool = True) -> pd.DataFrame:
    api_key = _api_key()
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())

    all_rows: list[dict] = []
    cursor = end_ts
    if resume and CHECKPOINT_PATH.exists():
        existing = pd.read_parquet(CHECKPOINT_PATH)
        all_rows = existing.to_dict("records")
        cursor = int(pd.to_datetime(existing["published_at"]).min().timestamp()) - 1
        print(f"resuming: {len(all_rows)} rows cached, cursor at {pd.to_datetime(cursor, unit='s')}")

    page = 0
    print(f"fetching CryptoCompare BTC news {start} -> {end}")
    while cursor > start_ts:
        articles = _fetch_page(cursor, api_key)
        if not articles:
            break
        for a in articles:
            pub_ts = a.get("published_on", 0)
            if pub_ts < start_ts:
                cursor = start_ts
                break
            all_rows.append({
                "published_at": pd.to_datetime(pub_ts, unit="s", utc=True),
                "title": a.get("title", ""),
                "body": a.get("body", ""),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
            })
            cursor = min(cursor, pub_ts)
        page += 1
        if page % 20 == 0:
            _checkpoint(all_rows)
            print(f"  checkpoint: {len(all_rows)} rows, at {pd.to_datetime(cursor, unit='s')}")
        time.sleep(0.2)

    df = _checkpoint(all_rows)
    if df.empty:
        return df

    df = df[(df["published_at"] >= pd.Timestamp(start)) & (df["published_at"] < pd.Timestamp(end))]
    print(f"done: {len(df)} total articles -> {CHECKPOINT_PATH}")
    return df.reset_index(drop=True)


def run_news(force_refresh: bool = False) -> pd.DataFrame:
    if CHECKPOINT_PATH.exists() and not force_refresh:
        print("Loading cached news (pass force_refresh=True to backfill further)")
        df = pd.read_parquet(CHECKPOINT_PATH)
        return df[(df["published_at"] >= pd.Timestamp(DATA_START)) &
                  (df["published_at"] < pd.Timestamp(DATA_END))].reset_index(drop=True)
    return collect_btc_news()


if __name__ == "__main__":
    df = run_news()
    print(df.tail())
    print(df.shape)
