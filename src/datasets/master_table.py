"""
Stage 9 — master table. One row per Binance hour. Orchestrates every
src.data.* / src.features.* module and merges them onto the hourly BTC
spot timeline with pd.merge_asof(direction="backward") wherever a source
has a coarser cadence (daily on-chain, daily macro, 8h funding) — that
guarantees every merged value was actually known at or before
`timestamp`, which is the leakage-safe condition Stage 4/5 call out
("use only the latest observation actually known by prediction_time").

Toggle feature families in configs/approach1.yaml `feature_families`.
Sources needing a key you haven't set yet degrade to "not merged" with a
log line, rather than failing the whole pipeline — see Stage 14 (ablation):
you're expected to run this with a growing feature set over time.
"""
from __future__ import annotations

import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger("datasets.master_table")


def _merge_asof_backward(base: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    if other is None or other.empty:
        log.warning(f"skipping merge of '{label}' — empty/unavailable")
        return base
    other = other.sort_values("timestamp")
    return pd.merge_asof(base.sort_values("timestamp"), other, on="timestamp", direction="backward")


def build_master_table(start: str = "2019-01-01", end: str | None = None) -> pd.DataFrame:
    from src.data import binance_futures, binance_spot, fear_greed, macro, news, onchain
    from src.datasets.labels import add_target
    from src.datasets.sequences import assert_no_gaps
    from src.features import derivatives, microstructure, news_features, onchain_features, technical

    cfg = load_config("approach1.yaml")
    families = cfg["feature_families"]
    symbol = cfg["symbol"]

    log.info("Stage 1/2 — spot + microstructure")
    spot = binance_spot.load_or_download(symbol=symbol, start=start, end=end)
    base = technical.build_technical_features(spot) if families.get("spot_technical") else spot
    if families.get("spot_order_flow"):
        base = microstructure.build_microstructure_features(base)
    assert_no_gaps(base)

    if families.get("derivatives"):
        log.info("Stage 3 — derivatives")
        funding = derivatives.build_funding_features(binance_futures.fetch_funding_rate(symbol, start, end))
        premium = derivatives.build_basis_features(binance_futures.fetch_premium_index_klines(symbol, "1h", start, end))
        oi = derivatives.build_oi_features(binance_futures.fetch_open_interest_hist(symbol))  # last ~30d only
        base = _merge_asof_backward(base, funding, "funding")
        base = _merge_asof_backward(base, premium, "premium/basis")
        base = _merge_asof_backward(base, oi, "open_interest")
        base = derivatives.add_interaction_features(base)

    if families.get("onchain"):
        log.info("Stage 4 — on-chain")
        oc = onchain_features.build_onchain_features(onchain.fetch_onchain_features(start=start, end=end))
        base = _merge_asof_backward(base, oc, "onchain")

    if families.get("fear_greed"):
        log.info("Stage 5 — Fear & Greed")
        fng = fear_greed.load_or_download()
        base = _merge_asof_backward(base, fng[["timestamp", "fear_greed"]], "fear_greed")
        base["fng_change_1d"] = base["fear_greed"].diff(24)
        base["fng_change_3d"] = base["fear_greed"].diff(72)
        base["fng_mean_7d"] = base["fear_greed"].rolling(168).mean()
        roll = base["fear_greed"].rolling(720)  # ~30d
        base["fng_zscore"] = (base["fear_greed"] - roll.mean()) / roll.std()

    if families.get("news_finbert") or families.get("news_llm_events"):
        log.info("Stage 6/7 — news")
        raw_news = news.load_cached()
        if raw_news.empty:
            log.warning("no cached news (run `python -m src.data.news` first with CC_API_KEY set) — skipping")
        else:
            timeline = base["timestamp"]
            if families.get("news_finbert"):
                scored = news_features.score_finbert_sentiment(raw_news)
                sent_feats = news_features.aggregate_sentiment(scored, timeline)
                base = base.merge(sent_feats, on="timestamp", how="left")
            if families.get("news_llm_events"):
                events = news_features.classify_news_events(raw_news)
                if "event_type" in events.columns:
                    event_feats = news_features.aggregate_event_features(events, timeline)
                    base = base.merge(event_feats, on="timestamp", how="left")

    if families.get("macro"):
        log.info("Stage 8 — macro")
        mac = macro.load_or_download(start=start, end=end)
        base = _merge_asof_backward(base, mac, "macro")

    log.info("Stage 9 — target")
    base = add_target(base)

    log.info(f"master table: {base.shape[0]} rows x {base.shape[1]} columns "
             f"({base['timestamp'].min()} -> {base['timestamp'].max()})")
    return base


if __name__ == "__main__":
    df = build_master_table()
    out_path = "data/processed/master_table.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved -> {out_path}")
    print(df.tail())
