"""
Target definition — the one place this gets defined, everything else
imports from here.

    Input:  everything known at or before t (prediction_time)
    Target: is Binance BTCUSDT spot close(t + 24h) > close(t) ?
"""
from __future__ import annotations

import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger("datasets.labels")


def add_target(hourly_df: pd.DataFrame, horizon_hours: int | None = None) -> pd.DataFrame:
    """
    hourly_df: must be sorted ascending by `timestamp`, one row per hour, with
    a `close` column, and no gaps (see src.datasets.sequences.assert_no_gaps).

    Adds: future_return_24h, target, label_end_time.
    """
    if horizon_hours is None:
        horizon_hours = load_config("approach1.yaml")["target"]["horizon_hours"]

    df = hourly_df.sort_values("timestamp").reset_index(drop=True).copy()
    df["future_return_24h"] = df["close"].shift(-horizon_hours) / df["close"] - 1
    df["target"] = (df["future_return_24h"] > 0).astype("Int64")
    df.loc[df["future_return_24h"].isna(), "target"] = pd.NA
    df["label_end_time"] = df["timestamp"] + pd.Timedelta(hours=horizon_hours)

    n_labeled = df["target"].notna().sum()
    log.info(f"labeled {n_labeled}/{len(df)} rows "
             f"(last {horizon_hours} rows are unlabeled — no future close yet)")
    return df
