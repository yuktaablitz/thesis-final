"""Stage 2 — spot order flow, derived directly from Binance kline fields."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("features.microstructure")

IMBALANCE_MEAN_WINDOWS = [3, 6, 12, 24]


def build_microstructure_features(spot_df: pd.DataFrame) -> pd.DataFrame:
    """spot_df must have: volume, taker_buy_base, num_trades, quote_asset_volume."""
    df = spot_df.copy()

    df["taker_buy_ratio"] = df["taker_buy_base"] / df["volume"]
    df["taker_sell_ratio"] = 1 - df["taker_buy_ratio"]
    df["taker_imbalance_1h"] = df["taker_buy_ratio"] - df["taker_sell_ratio"]

    for w in IMBALANCE_MEAN_WINDOWS:
        df[f"taker_imbalance_mean_{w}h"] = df["taker_imbalance_1h"].rolling(w).mean()

    df["trade_count_change"] = df["num_trades"].pct_change()
    df["average_trade_size"] = df["quote_asset_volume"] / df["num_trades"].replace(0, np.nan)
    df["quote_volume_change"] = df["quote_asset_volume"].pct_change()

    df = df.replace([np.inf, -np.inf], np.nan)
    log.info("built microstructure feature columns")
    return df
