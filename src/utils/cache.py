"""
On-disk parquet cache for any downloader. Every src/data/*.py module should
go through this so re-running a notebook never re-hits an API for data it
already has, and so the same cached file can be reused from local VS Code
and from Colab (just point CACHE_DIR at the same synced folder, e.g. Drive).

Usage:
    from src.utils.cache import cached_fetch

    def _fetch(start, end):
        ...  # real API call, returns a DataFrame
        return df

    df = cached_fetch("binance_spot_BTCUSDT_1h", _fetch, start="2019-01-01", end="2026-01-01")
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.utils.logging import get_logger

log = get_logger("cache")

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _key(name: str, kwargs: dict[str, Any]) -> str:
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{name}__{digest}"


def cache_path(name: str, **kwargs: Any) -> Path:
    return CACHE_DIR / f"{_key(name, kwargs)}.parquet"


def cached_fetch(
    name: str,
    fetch_fn: Callable[..., pd.DataFrame],
    force_refresh: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Cache `fetch_fn(**kwargs)` (which must return a DataFrame) to
    data/cache/<name>__<hash of kwargs>.parquet. Set force_refresh=True to
    bypass and overwrite. Raw/append-only sources (e.g. incremental news
    collectors) should manage their own checkpoint file instead — see
    src/data/news.py — and call this only for the final assembled frame.
    """
    path = cache_path(name, **kwargs)
    if path.exists() and not force_refresh:
        log.info(f"cache hit  {name} -> {path.name}")
        return pd.read_parquet(path)

    log.info(f"cache miss {name} -> fetching")
    df = fetch_fn(**kwargs)
    df.to_parquet(path, index=False)
    log.info(f"cached {name} ({len(df)} rows) -> {path.name}")
    return df
