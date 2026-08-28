"""
Stage 6 (FinBERT sentiment) + Stage 7 (LLM event classification) on top of
src.data.news's cached raw articles.

FinBERT needs no API key (public HF model, ProsusAI/finbert) — runs locally
on CPU/MPS but is meaningfully faster on Colab GPU for a full backfill.

LLM event classification needs ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.
The LLM sees only headline/body text — never future price — so this stays
leakage-safe (see design doc Stage 7). Every classified article is cached
to disk by URL; re-running this module NEVER re-calls the LLM for an
article it has already classified.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.utils.config import load_config
from src.utils.logging import get_logger

load_dotenv()
log = get_logger("features.news")

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
FINBERT_CACHE = CACHE_DIR / "finbert_sentiment.parquet"
LLM_CACHE = CACHE_DIR / "llm_event_cache.jsonl"

AGG_WINDOWS = ["1h", "3h", "6h", "12h", "24h"]


# ---------------------------------------------------------------- FinBERT --
def _load_finbert_pipeline():
    from transformers import pipeline
    log.info("loading FinBERT (ProsusAI/finbert) — first call downloads the model")
    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)


def score_finbert_sentiment(news_df: pd.DataFrame, text_col: str = "title",
                             batch_size: int = 32, force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns news_df + [positive, negative, neutral, sentiment_score], cached
    by URL so a re-run only scores newly-collected articles.
    """
    if news_df.empty:
        return news_df

    cached = pd.read_parquet(FINBERT_CACHE) if (FINBERT_CACHE.exists() and not force_refresh) else \
        pd.DataFrame(columns=["url"])
    todo = news_df[~news_df["url"].isin(cached["url"])].copy()
    log.info(f"FinBERT: {len(cached)} cached, {len(todo)} new articles to score")

    if not todo.empty:
        clf = _load_finbert_pipeline()
        texts = todo[text_col].fillna("").tolist()
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            out = clf(batch)
            results.extend(out)
            if i % (batch_size * 20) == 0:
                log.info(f"  scored {i + len(batch)}/{len(texts)}")

        rows = []
        for scores in results:
            by_label = {d["label"].lower(): d["score"] for d in scores}
            rows.append({
                "positive": by_label.get("positive", np.nan),
                "negative": by_label.get("negative", np.nan),
                "neutral": by_label.get("neutral", np.nan),
            })
        scored = pd.DataFrame(rows)
        todo = pd.concat([todo.reset_index(drop=True), scored], axis=1)
        todo["sentiment_score"] = todo["positive"] - todo["negative"]
        cached = pd.concat([cached, todo], ignore_index=True) if not cached.empty else todo
        cached.to_parquet(FINBERT_CACHE, index=False)

    merged = news_df.merge(
        cached[["url", "positive", "negative", "neutral", "sentiment_score"]], on="url", how="left"
    )
    return merged


def aggregate_sentiment(scored_df: pd.DataFrame, timeline: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Rolling aggregation of scored articles onto an hourly `timeline`, using
    only articles with timestamp <= each timeline hour (point-in-time safe).
    """
    scored_df = scored_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = pd.DataFrame({"timestamp": timeline})

    s = scored_df.set_index("timestamp")
    for w in AGG_WINDOWS:
        counts, pos_mean, neg_mean, sent_mean, sent_std = [], [], [], [], []
        window = pd.Timedelta(w)
        # Efficient-enough for hourly cadence over a few years; swap for a
        # merge_asof + rolling window trick if this becomes a bottleneck.
        for ts in out["timestamp"]:
            slice_ = s.loc[(s.index > ts - window) & (s.index <= ts)]
            counts.append(len(slice_))
            pos_mean.append(slice_["positive"].mean())
            neg_mean.append(slice_["negative"].mean())
            sent_mean.append(slice_["sentiment_score"].mean())
            sent_std.append(slice_["sentiment_score"].std())
        out[f"news_count_{w}"] = counts
        out[f"positive_mean_{w}"] = pos_mean
        out[f"negative_mean_{w}"] = neg_mean
        out[f"sentiment_mean_{w}"] = sent_mean
        out[f"sentiment_std_{w}"] = sent_std

    return out


# ------------------------------------------------------------- LLM events --
EVENT_PROMPT_TEMPLATE = """You classify a single Bitcoin-related news headline/body into a
structured event record. Respond with ONLY a JSON object, no prose, matching this schema:

{{"event_type": one of {event_types}, "impact_direction": "bullish"|"bearish"|"neutral",
 "impact_strength": integer 1-5}}

Headline: {title}
Body: {body}
"""


def _load_llm_cache() -> dict:
    if not LLM_CACHE.exists():
        return {}
    cache = {}
    with open(LLM_CACHE) as f:
        for line in f:
            rec = json.loads(line)
            cache[rec["url"]] = rec
    return cache


def _append_llm_cache(rec: dict) -> None:
    with open(LLM_CACHE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _classify_one(title: str, body: str, event_types: list[str], client_kind: str, client) -> dict:
    prompt = EVENT_PROMPT_TEMPLATE.format(
        event_types=event_types, title=title, body=(body or "")[:2000]
    )
    if client_kind == "anthropic":
        resp = client.messages.create(
            model="claude-3-5-haiku-latest", max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
    else:  # openai
        resp = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content

    text = text.strip().strip("`").removeprefix("json").strip()
    return json.loads(text)


def classify_news_events(news_df: pd.DataFrame, max_articles: int | None = None) -> pd.DataFrame:
    """
    Classifies each article's event_type/impact_direction/impact_strength via
    an LLM, using and updating data/cache/llm_event_cache.jsonl so every
    article is sent to the LLM at most once, ever.
    """
    cfg = load_config("approach1.yaml")
    event_types = cfg["news_event_types"]

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not anthropic_key and not openai_key:
        log.warning("no ANTHROPIC_API_KEY or OPENAI_API_KEY set — skipping LLM event classification")
        return news_df

    if anthropic_key:
        import anthropic
        client, client_kind = anthropic.Anthropic(api_key=anthropic_key), "anthropic"
    else:
        import openai
        client, client_kind = openai.OpenAI(api_key=openai_key), "openai"

    cache = _load_llm_cache()
    todo = news_df[~news_df["url"].isin(cache.keys())]
    if max_articles:
        todo = todo.head(max_articles)
    log.info(f"LLM events: {len(cache)} cached, classifying {len(todo)} new articles via {client_kind}")

    for _, row in todo.iterrows():
        try:
            result = _classify_one(row["title"], row.get("body", ""), event_types, client_kind, client)
            rec = {"url": row["url"], **result}
        except Exception as e:
            log.warning(f"LLM classify failed for {row['url']}: {e}")
            rec = {"url": row["url"], "event_type": "OTHER", "impact_direction": "neutral",
                   "impact_strength": 1}
        cache[rec["url"]] = rec
        _append_llm_cache(rec)

    events_df = pd.DataFrame(cache.values())
    return news_df.merge(events_df, on="url", how="left")


def aggregate_event_features(events_df: pd.DataFrame, timeline: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Rolling counts like ETF_bullish_count_24h, hack_count_6h, etc. — one
    column per (event_type, direction, window) combo the design doc names as
    examples; extend WINDOWS_BY_TYPE below as needed.
    """
    events_df = events_df.dropna(subset=["timestamp", "event_type"]).sort_values("timestamp")
    s = events_df.set_index("timestamp")
    out = pd.DataFrame({"timestamp": timeline})

    combos = [
        ("ETF", "bullish", "24h"), ("REGULATION", "bearish", "24h"),
        ("HACK_SECURITY", None, "6h"), ("INSTITUTIONAL", "bullish", "12h"),
        ("MACRO", "bearish", "24h"),
    ]
    for event_type, direction, w in combos:
        window = pd.Timedelta(w)
        col = f"{event_type.lower()}_{direction + '_' if direction else ''}count_{w}"
        counts = []
        for ts in out["timestamp"]:
            slice_ = s.loc[(s.index > ts - window) & (s.index <= ts)]
            slice_ = slice_[slice_["event_type"] == event_type]
            if direction:
                slice_ = slice_[slice_["impact_direction"] == direction]
            counts.append(len(slice_))
        out[col] = counts
    return out
