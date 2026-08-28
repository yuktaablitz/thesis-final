"""
News feature engineering, two stages:
  1. FinBERT sentiment (no key — public HF model, ProsusAI/finbert). Runs
     on MPS locally, CUDA on Colab, or CPU. Cached to data/cache/ so a
     re-run only scores newly-collected articles.
  2. LLM event classification (needs DEEPSEEK_API_KEY, or ANTHROPIC_API_KEY
     / OPENAI_API_KEY as a fallback) — runs AFTER FinBERT, on the FinBERT
     output, per the design doc ("LLM event extraction comes after
     FinBERT"). Every classified article is cached by URL and never
     re-sent to the LLM.

Hourly aggregation buckets each article to its published hour
(`published_at.dt.floor("h")`) and merges by equality onto the master
clock — see src/master_table.py for why this is treated as point-in-time
consistent with how `close` itself is used there.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CACHE_DIR
from src.common import save_parquet

load_dotenv()

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_CACHE = CACHE_DIR / "news_finbert.parquet"
LLM_CACHE_PATH = CACHE_DIR / "news_llm_events.jsonl"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

EVENT_TYPES = [
    "ETF", "REGULATION", "MACRO", "FED", "HACK_SECURITY", "EXCHANGE",
    "INSTITUTIONAL", "MINING", "HALVING", "LIQUIDATION", "WHALE",
    "ADOPTION", "TECHNOLOGY", "OTHER",
]


# ---------------------------------------------------------------- FinBERT --
def score_finbert(news: pd.DataFrame, batch_size: int = 64, force_refresh: bool = False) -> pd.DataFrame:
    """Adds news_negative/news_neutral/news_positive/news_sentiment, cached by URL."""
    if news.empty:
        return news

    cached = pd.read_parquet(FINBERT_CACHE) if (FINBERT_CACHE.exists() and not force_refresh) \
        else pd.DataFrame(columns=["url"])
    todo = news[~news["url"].isin(cached["url"])].copy()
    print(f"FinBERT: {len(cached)} cached, {len(todo)} new articles to score")

    if not todo.empty:
        device = ("mps" if torch.backends.mps.is_available()
                   else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"FinBERT device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(device)
        model.eval()

        texts = todo["title"].fillna("").astype(str).tolist()
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                logits = model(**tokens).logits
            prob = torch.softmax(logits, dim=1)
            outputs.append(prob.cpu().numpy())
            if start % (batch_size * 20) == 0:
                print(f"  scored {start + len(batch)}/{len(texts)}")

        p = np.vstack(outputs)
        # FinBERT label ordering: negative, neutral, positive
        todo = todo.reset_index(drop=True)
        todo["news_negative"] = p[:, 0]
        todo["news_neutral"] = p[:, 1]
        todo["news_positive"] = p[:, 2]
        todo["news_sentiment"] = todo["news_positive"] - todo["news_negative"]

        cached = pd.concat([cached, todo], ignore_index=True) if not cached.empty else todo
        save_parquet(cached, FINBERT_CACHE)

    score_cols = ["url", "news_negative", "news_neutral", "news_positive", "news_sentiment"]
    return news.merge(cached[score_cols], on="url", how="left")


def aggregate_news_hourly(news: pd.DataFrame) -> pd.DataFrame:
    """One row per hour an article was published in, with mean/count sentiment stats."""
    df = news.dropna(subset=["published_at"]).copy()
    df["timestamp"] = df["published_at"].dt.floor("h")

    hourly = (
        df.groupby("timestamp")
        .agg(
            news_count=("title", "count"),
            news_positive=("news_positive", "mean"),
            news_negative=("news_negative", "mean"),
            news_neutral=("news_neutral", "mean"),
            news_sentiment=("news_sentiment", "mean"),
            news_sentiment_std=("news_sentiment", "std"),
        )
        .reset_index()
    )
    hourly["news_sentiment_std"] = hourly["news_sentiment_std"].fillna(0)
    return hourly


def add_news_rollups(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing rolling windows on the hourly-aggregated news frame."""
    df = df.set_index("timestamp").sort_index()
    for h in [3, 6, 12, 24]:
        df[f"news_count_{h}h"] = df["news_count"].rolling(f"{h}h").sum()
        df[f"news_sentiment_{h}h"] = df["news_sentiment"].rolling(f"{h}h").mean()
    return df.reset_index()


# ------------------------------------------------------------- LLM events --
EVENT_PROMPT = """You classify a single Bitcoin-related news headline into a structured
event record. Respond with ONLY a JSON object, no prose, matching this schema:

{{"event_type": one of {event_types}, "impact_direction": "bullish"|"bearish"|"neutral",
 "impact_strength": integer 1-5}}

Headline: {title}
"""


def _load_llm_cache() -> dict:
    if not LLM_CACHE_PATH.exists():
        return {}
    cache = {}
    with open(LLM_CACHE_PATH) as f:
        for line in f:
            rec = json.loads(line)
            cache[rec["url"]] = rec
    return cache


def _append_llm_cache(rec: dict) -> None:
    LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LLM_CACHE_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _classify_one(title: str, client_kind: str, client) -> dict:
    prompt = EVENT_PROMPT.format(event_types=EVENT_TYPES, title=title)
    if client_kind == "anthropic":
        resp = client.messages.create(model="claude-3-5-haiku-latest", max_tokens=150,
                                       messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text
    else:  # "deepseek" or "openai" — both OpenAI-compatible chat.completions
        resp = client.chat.completions.create(
            model="deepseek-chat" if client_kind == "deepseek" else "gpt-4o-mini",
            max_tokens=150, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content
    text = text.strip().strip("`").removeprefix("json").strip()
    return json.loads(text)


def classify_news_events(news: pd.DataFrame, max_articles: int | None = None) -> pd.DataFrame:
    """
    Provider priority: DEEPSEEK_API_KEY > ANTHROPIC_API_KEY > OPENAI_API_KEY.
    Run this AFTER score_finbert — cheap enough to run on the full backfill,
    but still only ever hits the LLM once per article, ever.
    """
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not (deepseek_key or anthropic_key or openai_key):
        print("no DEEPSEEK_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY set — skipping LLM event classification")
        return news

    if deepseek_key:
        import openai
        client, client_kind = openai.OpenAI(api_key=deepseek_key, base_url=DEEPSEEK_BASE_URL), "deepseek"
    elif anthropic_key:
        import anthropic
        client, client_kind = anthropic.Anthropic(api_key=anthropic_key), "anthropic"
    else:
        import openai
        client, client_kind = openai.OpenAI(api_key=openai_key), "openai"

    cache = _load_llm_cache()
    todo = news[~news["url"].isin(cache.keys())]
    if max_articles:
        todo = todo.head(max_articles)
    print(f"LLM events: {len(cache)} cached, classifying {len(todo)} new articles via {client_kind}")

    for _, row in todo.iterrows():
        try:
            result = _classify_one(row["title"], client_kind, client)
            rec = {"url": row["url"], **result}
        except Exception as e:
            print(f"LLM classify failed for {row['url']}: {e}")
            rec = {"url": row["url"], "event_type": "OTHER", "impact_direction": "neutral", "impact_strength": 1}
        cache[rec["url"]] = rec
        _append_llm_cache(rec)

    events_df = pd.DataFrame(cache.values())
    return news.merge(events_df, on="url", how="left")


def aggregate_event_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Hourly rolling counts for a handful of named event/direction/window combos.
    Builds a unique-per-hour count series first (groupby, not a duplicate-
    timestamp rolling) so the result can be equality-merged onto the master
    clock without a many-to-one fan-out.
    """
    df = events.dropna(subset=["published_at", "event_type"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = df["published_at"].dt.floor("h")
    full_index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="1h", tz="UTC")

    combos = [
        ("ETF", "bullish", "24h"), ("REGULATION", "bearish", "24h"),
        ("HACK_SECURITY", None, "6h"), ("INSTITUTIONAL", "bullish", "12h"),
        ("MACRO", "bearish", "24h"),
    ]
    out = None
    for event_type, direction, w in combos:
        match = df["event_type"] == event_type
        if direction:
            match &= df["impact_direction"] == direction
        col = f"{event_type.lower()}_{(direction + '_') if direction else ''}count_{w}"

        hourly_counts = df.loc[match].groupby("timestamp").size().reindex(full_index, fill_value=0)
        hourly_counts.index.name = "timestamp"
        rolled = hourly_counts.rolling(w).sum().reset_index()
        rolled.columns = ["timestamp", col]
        out = rolled if out is None else out.merge(rolled, on="timestamp", how="outer")

    return out.sort_values("timestamp").reset_index(drop=True) if out is not None else pd.DataFrame()
