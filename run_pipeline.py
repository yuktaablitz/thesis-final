"""
End-to-end Approach 1 data pipeline: download every source (cached — never
recompute what's already on disk), build the leakage-safe hourly master
table, generate STEP=6 primary samples, split, and run the leakage audits.

    python run_pipeline.py                  # full pipeline, primary STEP=6
    python run_pipeline.py --step 24         # benchmark cadence
    python run_pipeline.py --skip-news       # skip CryptoCompare/FinBERT/LLM (no CC_API_KEY yet)
    python run_pipeline.py --skip-llm-events # FinBERT sentiment only, skip DeepSeek/Anthropic/OpenAI

Do not build XGBoost/TCN/CNN-LSTM models against this until the printed
coverage report and leakage audits below look right — see README.
"""
import argparse

import pandas as pd

from config import PROCESSED_DIR, RESULTS_DIR, STEP_HOURS
from src.binance_futures import add_derivatives_features, run_funding, run_futures_metrics
from src.binance_spot import run as run_spot
from src.common import coverage_report
from src.fear_greed import run_fng
from src.leakage import assert_no_label_overlap, assert_point_in_time
from src.master_table import build_master, report_modalities
from src.news import run_news
from src.news_features import (
    add_news_rollups, aggregate_event_features, aggregate_news_hourly,
    classify_news_events, score_finbert,
)
from src.onchain import run_onchain
from src.samples import create_target, select_prediction_times, split_samples
from src.technical_features import add_market_features


def main(step_hours: int = STEP_HOURS, skip_news: bool = False, skip_llm_events: bool = False,
         force_refresh: bool = False) -> None:
    # ============================================================ 1. Binance BTC Spot
    spot = run_spot(force_refresh=force_refresh)
    spot = add_market_features(spot)
    coverage_report("spot+technical", spot)

    # ============================================================ 2. Binance derivatives
    funding = run_funding(force_refresh=force_refresh)
    futures_metrics = run_futures_metrics(force_refresh=force_refresh)
    coverage_report("funding", funding) if not funding.empty else print("funding: EMPTY")
    coverage_report("futures_metrics", futures_metrics) if not futures_metrics.empty else print("futures_metrics: EMPTY")

    # ============================================================ 3. Fear & Greed
    fng = run_fng(force_refresh=force_refresh)
    coverage_report("fear_greed", fng, timestamp_col="source_time")

    # ============================================================ 4. News (FinBERT + LLM events)
    news_hourly = None
    if not skip_news:
        try:
            news = run_news(force_refresh=force_refresh)
            if not news.empty:
                news_scored = score_finbert(news)
                if not skip_llm_events:
                    news_scored = classify_news_events(news_scored)
                    events_hourly = aggregate_event_features(news_scored)
                else:
                    events_hourly = None
                news_hourly = add_news_rollups(aggregate_news_hourly(news_scored))
                if events_hourly is not None and not events_hourly.empty:
                    news_hourly = news_hourly.merge(events_hourly, on="timestamp", how="left")
                coverage_report("news", news_hourly)
        except RuntimeError as e:
            print(f"[skip] news: {e}")

    # ============================================================ 5. On-chain
    onchain = run_onchain(force_refresh=force_refresh)
    if not onchain.empty:
        coverage_report("onchain", onchain, timestamp_col="source_time")

    # ============================================================ 6. Macro (FRED)
    from src.macro import run_macro
    macro_frames = run_macro(force_refresh=force_refresh)

    # ============================================================ 7. Master table
    master = build_master(
        spot=spot, funding=funding, futures_metrics=futures_metrics, fng=fng,
        news_hourly=news_hourly, onchain=onchain, macro_frames=macro_frames,
    )
    master = add_derivatives_features(master)
    coverage_report("master (pre-target)", master)

    # ============================================================ 8. Target
    master = create_target(master)

    # ============================================================ 9. Coverage audit -> results/data_coverage.csv
    coverage_df = report_modalities(master)
    coverage_df.to_csv(RESULTS_DIR / "data_coverage.csv", index=False)

    # ============================================================ 10. Point-in-time audit (fails loudly)
    assert_point_in_time(master)

    master.to_parquet(PROCESSED_DIR / "master_table.parquet", index=False)
    print(f"master table saved -> {PROCESSED_DIR / 'master_table.parquet'} ({master.shape[0]} x {master.shape[1]})")

    # ============================================================ 11. Sample selection + split
    samples = select_prediction_times(master, step_hours=step_hours)
    train, val, test = split_samples(samples)

    # ============================================================ 12. Leakage audit (fails loudly)
    assert_no_label_overlap(train, val, test)

    print(f"\nSTEP={step_hours}h  TRAIN: {train.shape}  VAL: {val.shape}  TEST: {test.shape}")
    for name, split_df in (("train", train), ("validation", val), ("test", test)):
        if len(split_df):
            print(f"  {name} class balance: {split_df['target'].astype(int).mean():.3f} positive "
                  f"(n={len(split_df)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=STEP_HOURS, help="sampling cadence in hours (6 primary, 24 benchmark, 1 sensitivity)")
    ap.add_argument("--skip-news", action="store_true", help="skip CryptoCompare/FinBERT/LLM entirely (e.g. no CC_API_KEY yet)")
    ap.add_argument("--skip-llm-events", action="store_true", help="run FinBERT sentiment but skip the LLM event classifier")
    ap.add_argument("--force-refresh", action="store_true", help="ignore all caches and re-download everything")
    args = ap.parse_args()
    main(step_hours=args.step, skip_news=args.skip_news, skip_llm_events=args.skip_llm_events,
         force_refresh=args.force_refresh)
