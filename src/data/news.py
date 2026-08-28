"""
Stage 6 — BTC news collection via CryptoCompare. Requires CC_API_KEY in
.env (get one free at https://www.cryptocompare.com/cryptopian/api-keys).

Adapted from the Semester 1 notebook's collector, with two changes:
  1. The key is read from the environment, never hardcoded — the notebook
     version had a real key pasted into a cell; that key must be rotated
     (see README "Security note") and this file must never regain one.
  2. Same resumable-checkpoint pattern (walks backward in time via `lTs`,
     writes to disk every 20 pages) so a long backfill survives interruption.

This module only fetches raw articles. Sentiment (FinBERT) and event
classification (LLM) happen in src/features/news_features.py — kept
separate so raw articles are cached exactly once and never re-fetched.
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
log = get_logger("news")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RAW_DIR / "cryptocompare_news.parquet"


def _get_api_key() -> str:
    key = os.environ.get("CC_API_KEY")
    if not key:
        raise RuntimeError(
            "CC_API_KEY not set. Copy .env.example to .env and add a CryptoCompare key: "
            "https://www.cryptocompare.com/cryptopian/api-keys"
        )
    return key


def _fetch_page(before_ts: int, api_key: str, retries: int = 5) -> list[dict]:
    params = {"lang": "EN", "categories": "BTC", "lTs": before_ts, "api_key": api_key}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get("https://min-api.cryptocompare.com/data/v2/news/",
                              params=params, timeout=45)
            r.raise_for_status()
            return r.json().get("Data", [])
        except Exception as e:
            log.warning(f"attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 ** attempt)
    return []


def collect_btc_news(start: str = "2019-01-01", end: str | None = None,
                      resume: bool = True) -> pd.DataFrame:
    """Backfills CryptoCompare BTC news, checkpointing to disk every 20 pages."""
    api_key = _get_api_key()
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())

    all_rows: list[dict] = []
    cursor = end_ts
    if resume and CHECKPOINT_PATH.exists():
        existing = pd.read_parquet(CHECKPOINT_PATH)
        all_rows = existing.to_dict("records")
        cursor = int(pd.to_datetime(existing["timestamp"]).min().timestamp()) - 1
        log.info(f"resuming: {len(all_rows)} rows cached, cursor at {pd.to_datetime(cursor, unit='s')}")

    page = 0
    log.info(f"fetching CryptoCompare BTC news {start} -> {end}")
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
                "timestamp": pd.to_datetime(pub_ts, unit="s", utc=True),
                "title": a.get("title", ""),
                "body": a.get("body", ""),
                "source": a.get("source", ""),
                "categories": a.get("categories", ""),
                "url": a.get("url", ""),
            })
            cursor = min(cursor, pub_ts)
        page += 1
        if page % 20 == 0:
            _checkpoint(all_rows)
            log.info(f"  checkpoint: {len(all_rows)} rows, at {pd.to_datetime(cursor, unit='s')}")
        time.sleep(0.2)

    df = _checkpoint(all_rows)
    log.info(f"done: {len(df)} total articles -> {CHECKPOINT_PATH}")
    return df


def _checkpoint(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset="url").sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(CHECKPOINT_PATH, index=False)
    return df


def load_cached() -> pd.DataFrame:
    if not CHECKPOINT_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "title", "body", "source", "categories", "url"])
    return pd.read_parquet(CHECKPOINT_PATH)


if __name__ == "__main__":
    df = collect_btc_news()
    print(df.tail())
    print(df.shape)
