"""
Stage 3 — Binance BTC derivatives features: funding, open interest, basis,
and their interactions with spot returns. Consumes the raw frames produced
by src.data.binance_futures and, for the interaction terms, spot returns
from src.features.technical.

Point-in-time note: funding/premium-index are merged with merge_asof
(backward) against the 1h spot timeline in src.datasets.master_table, not
here — this module only computes the derived columns on the source's own
timeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("features.derivatives")


def build_funding_features(funding_df: pd.DataFrame) -> pd.DataFrame:
    """funding_df: [timestamp, funding_rate] at Binance's native 8h cadence."""
    df = funding_df.sort_values("timestamp").reset_index(drop=True).copy()
    df["funding_mean_24h"] = df["funding_rate"].rolling(3).mean()   # ~3 fundings = 24h
    df["funding_mean_72h"] = df["funding_rate"].rolling(9).mean()   # ~9 fundings = 72h
    roll = df["funding_rate"].rolling(30)
    df["funding_zscore"] = (df["funding_rate"] - roll.mean()) / roll.std()
    return df


def build_basis_features(premium_df: pd.DataFrame) -> pd.DataFrame:
    """premium_df: [timestamp, premium_index] hourly (mark vs index premium)."""
    df = premium_df.sort_values("timestamp").reset_index(drop=True).copy()
    df = df.rename(columns={"premium_index": "perp_spot_basis"})
    df["basis_change"] = df["perp_spot_basis"].diff()
    roll = df["perp_spot_basis"].rolling(168)
    df["basis_zscore"] = (df["perp_spot_basis"] - roll.mean()) / roll.std()
    return df


def build_oi_features(oi_df: pd.DataFrame) -> pd.DataFrame:
    """oi_df: [timestamp, open_interest]. See binance_futures docstring re: ~30d retention."""
    if oi_df.empty:
        return oi_df
    df = oi_df.sort_values("timestamp").reset_index(drop=True).copy()
    df["oi_change_1h"] = df["open_interest"].pct_change(1)
    df["oi_change_6h"] = df["open_interest"].pct_change(6)
    df["oi_change_24h"] = df["open_interest"].pct_change(24)
    roll = df["open_interest"].rolling(168)
    df["oi_zscore"] = (df["open_interest"] - roll.mean()) / roll.std()
    return df


def add_interaction_features(master: pd.DataFrame) -> pd.DataFrame:
    """
    Call this AFTER merging spot returns + funding + OI into one hourly
    frame (see src.datasets.master_table). Requires columns: return_1h,
    return_24h, oi_change_1h, oi_change_24h, funding_rate.
    """
    df = master.copy()
    required = {"return_1h", "return_24h", "oi_change_1h", "oi_change_24h", "funding_rate"}
    missing = required - set(df.columns)
    if missing:
        log.warning(f"skipping interaction features, missing columns: {missing}")
        return df

    df["return1h_x_oichange1h"] = df["return_1h"] * df["oi_change_1h"]
    df["return24h_x_oichange24h"] = df["return_24h"] * df["oi_change_24h"]
    df["funding_x_oichange24h"] = df["funding_rate"] * df["oi_change_24h"]
    df["funding_x_return24h"] = df["funding_rate"] * df["return_24h"]

    df = df.replace([np.inf, -np.inf], np.nan)
    return df
