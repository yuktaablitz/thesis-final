"""Stage 1/2 — stationary spot technicals + order flow, computed on Binance Spot OHLCV+microstructure."""
import numpy as np
import pandas as pd
import ta


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()

    # ---- returns ----
    for h in [1, 3, 6, 12, 24, 48, 72, 168]:
        df[f"ret_{h}h"] = df["close"].pct_change(h)

    # ---- volatility ----
    ret1 = df["close"].pct_change()
    for h in [6, 12, 24, 72, 168]:
        df[f"vol_{h}h"] = ret1.rolling(h).std()

    # ---- moving averages, relative (stationary) ----
    for h in [10, 20, 50, 100, 200]:
        ma = df["close"].rolling(h).mean()
        df[f"close_ma{h}_rel"] = df["close"] / ma - 1

    # ---- RSI ----
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["rsi_change_6h"] = df["rsi"].diff(6)

    # ---- MACD, normalized by price ----
    m = ta.trend.MACD(df["close"])
    df["macd_rel"] = m.macd() / df["close"]
    df["macd_signal_rel"] = m.macd_signal() / df["close"]
    df["macd_hist_rel"] = m.macd_diff() / df["close"]

    # ---- Bollinger ----
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    upper, lower, middle = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    df["bb_position"] = (df["close"] - lower) / (upper - lower)
    df["bb_width"] = (upper - lower) / middle

    # ---- distance to rolling high/low ----
    for h in [24, 72, 168]:
        high = df["high"].rolling(h).max()
        low = df["low"].rolling(h).min()
        df[f"dist_high_{h}h"] = df["close"] / high - 1
        df[f"dist_low_{h}h"] = df["close"] / low - 1

    # ---- candle structure ----
    df["candle_body_rel"] = (df["close"] - df["open"]) / df["close"]
    df["range_rel"] = (df["high"] - df["low"]) / df["close"]
    df["upper_wick_rel"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_wick_rel"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

    # ---- volume ----
    for h in [6, 24, 72, 168]:
        rolling_volume = df["volume"].rolling(h).mean()
        df[f"volume_ratio_{h}h"] = df["volume"] / rolling_volume

    # ---- spot taker pressure (order flow) ----
    df["spot_taker_buy_ratio"] = df["taker_buy_base"] / df["volume"]
    df["spot_taker_sell_ratio"] = 1 - df["spot_taker_buy_ratio"]
    df["spot_taker_imbalance"] = 2 * df["spot_taker_buy_ratio"] - 1
    for h in [3, 6, 12, 24]:
        df[f"spot_taker_imbalance_mean_{h}h"] = df["spot_taker_imbalance"].rolling(h).mean()

    # ---- trade activity ----
    df["avg_trade_size"] = df["volume"] / df["num_trades"].replace(0, np.nan)
    df["trade_count_change_1h"] = df["num_trades"].pct_change()
    df["quote_volume_change_1h"] = df["quote_asset_volume"].pct_change()

    # ---- time of day / day of week, cyclical ----
    hour = df["timestamp"].dt.hour
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df.replace([np.inf, -np.inf], np.nan)
