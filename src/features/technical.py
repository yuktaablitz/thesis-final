"""
Stage 1 — spot technicals, computed on top of src.data.binance_spot output.
All levels are made stationary (ratios/differences, not raw MA/price) per
the design doc, since raw levels don't generalize across BTC's price range
from 2019 to today.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from src.utils.logging import get_logger

log = get_logger("features.technical")

RETURN_HORIZONS = [1, 3, 6, 12, 24, 48, 72, 168]
VOL_HORIZONS = [6, 12, 24, 72, 168]


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for h in RETURN_HORIZONS:
        df[f"return_{h}h"] = df["close"].pct_change(h)
    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "return_1h" not in df.columns:
        df["return_1h"] = df["close"].pct_change(1)
    for h in VOL_HORIZONS:
        df[f"volatility_{h}h"] = df["return_1h"].rolling(h).std()
    return df


def add_stationary_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    for w in (10, 20, 50):
        ma = close.rolling(w).mean()
        df[f"close_ma{w}"] = close / ma - 1

    rsi = ta.momentum.RSIIndicator(close).rsi()
    df["rsi"] = rsi
    df["rsi_change"] = rsi.diff()

    macd = ta.trend.MACD(close)
    df["macd_over_close"] = macd.macd() / close
    df["macd_signal_over_close"] = macd.macd_signal() / close
    df["macd_hist_over_close"] = macd.macd_diff() / close

    bb = ta.volatility.BollingerBands(close)
    bb_high, bb_low = bb.bollinger_hband(), bb.bollinger_lband()
    bb_width = bb_high - bb_low
    df["bollinger_position"] = ((close - bb_low) / bb_width).replace([np.inf, -np.inf], np.nan)
    df["bollinger_width"] = bb_width / close

    high_24 = df["high"].rolling(24).max()
    low_24 = df["low"].rolling(24).min()
    high_168 = df["high"].rolling(168).max()
    low_168 = df["low"].rolling(168).min()
    df["distance_from_24h_high"] = close / high_24 - 1
    df["distance_from_24h_low"] = close / low_24 - 1
    df["distance_from_168h_high"] = close / high_168 - 1
    df["distance_from_168h_low"] = close / low_168 - 1

    return df


def add_candle_structure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    df["body_over_close"] = (close - df["open"]) / close
    df["range_over_close"] = (df["high"] - df["low"]) / close
    df["upper_wick_over_close"] = (df["high"] - df[["open", "close"]].max(axis=1)) / close
    df["lower_wick_over_close"] = (df[["open", "close"]].min(axis=1) - df["low"]) / close
    return df


def build_technical_features(spot_df: pd.DataFrame) -> pd.DataFrame:
    """spot_df: output of src.data.binance_spot.load_or_download (has `timestamp`, OHLCV)."""
    df = spot_df.sort_values("timestamp").reset_index(drop=True)
    df = add_returns(df)
    df = add_volatility(df)
    df = add_stationary_technicals(df)
    df = add_candle_structure(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    log.info(f"built {df.shape[1] - spot_df.shape[1]} technical feature columns")
    return df
