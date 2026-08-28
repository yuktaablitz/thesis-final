"""
Stage 4 feature shaping on top of src.data.onchain's raw daily frame:
selects the configured tier1/2/3 columns and derives rolling z-scores for
any raw level metric that doesn't already have one (e.g. mvrv_zscore from
mvrv, if you're on the free CoinMetrics tier and only have the raw MVRV).
On-chain data is daily; keep it daily here — point-in-time merge (latest
observation known by prediction_time, no look-ahead) happens in
src.datasets.master_table, matching Stage 4's "use only the latest
observation actually known by prediction_time" rule.
"""
from __future__ import annotations

import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger("features.onchain")

ZSCORE_WINDOW_DAYS = 365


def build_onchain_features(onchain_df: pd.DataFrame) -> pd.DataFrame:
    if onchain_df.empty:
        return onchain_df

    cfg = load_config("approach1.yaml")
    wanted = cfg["onchain"]["tier1"] + cfg["onchain"]["tier2"] + cfg["onchain"]["tier3"]

    df = onchain_df.sort_values("timestamp").reset_index(drop=True).copy()
    available = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        log.info(f"on-chain columns not available from current sources: {missing}")

    keep = ["timestamp", *available]
    df = df[keep]

    for col in available:
        zscore_col = f"{col}_zscore"
        if zscore_col not in wanted:
            continue
        roll = df[col].rolling(ZSCORE_WINDOW_DAYS)
        df[zscore_col] = (df[col] - roll.mean()) / roll.std()

    return df
