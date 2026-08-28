Predicts the direction of Binance BTCUSDT spot close, exactly 24 hours ahead:

```
Input:  everything known at or before t
Target: is close(t + 24h) > close(t) ?
```

All BTC market and derivatives data comes from Binance — no yfinance.
Macro is FRED (small, fixed set). On-chain is CoinMetrics' free Community
API, discovered from its catalog rather than assumed. News is CryptoCompare
+ FinBERT + an LLM event classifier (DeepSeek by default).

## Architecture

```
RAW SOURCE
   |
normalize timestamp to UTC
   |
create "available_at"        <- when this observation was actually knowable
   |
cache source separately       (data/raw/, data/cache/)
   |
feature engineering within source
   |
merge_asof BACKWARD onto Binance Spot hourly clock (never bfill)
   |
master hourly table           (data/processed/master_table.parquet)
   |
label t -> t+24h
   |
sample every 6h (primary) / 24h (benchmark) / 1h (sensitivity only)
   |
split by label_end_time, not just prediction_time
```

Binance Spot 1h klines are the master clock. Every other source — funding,
futures metrics, Fear & Greed, news, on-chain, macro — carries its own
`source_time` (when the observation describes) and `available_at` (when it
was actually knowable) and gets attached via `pd.merge_asof(..., direction="backward")`
on `available_at`, never on `source_time` directly and never with `bfill()`.
`src/leakage.py`'s `assert_point_in_time` audits every `*_available_at`
column independently — see below.

Binance USD-M futures launched 2019-09-13, so derivatives legitimately
don't exist before then. Spot/F&G/on-chain rows for Jan-Aug 2019 are kept,
not dropped — `src/master_table.add_missing_indicators` adds `<col>_missing`
flags instead.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
cp .env.example .env   # fill in CC_API_KEY and DEEPSEEK_API_KEY (or ANTHROPIC/OPENAI)
```

## Run

```bash
python run_pipeline.py                  # full pipeline, primary STEP=6
python run_pipeline.py --step 24         # benchmark cadence
python run_pipeline.py --skip-news       # no CC_API_KEY yet
python run_pipeline.py --skip-llm-events # FinBERT only, skip the LLM classifier
```

This downloads (and caches — nothing expensive is ever recomputed) every
source, builds the master table, prints a coverage report per modality
(saved to `results/data_coverage.csv`), runs the point-in-time and
label-overlap leakage audits (both raise `AssertionError` and stop
execution if violated — a leaky pipeline never silently produces a
result), and prints train/validation/test shapes and class balance.

Local (VS Code + `.venv`, MPS on Apple Silicon) for downloading, caching,
feature engineering, XGBoost/LightGBM. Colab GPU for FinBERT batch scoring
and TCN/CNN-LSTM/attention training — `notebooks/` is meant to be opened
there directly. Never commit raw data, checkpoints, or `.env`.

## What needs an API key

| Source | Key needed? |
|---|---|
| Binance spot + futures (klines, funding, archived daily metrics) | No |
| Fear & Greed (alternative.me) | No |
| Macro (FRED CSV endpoint: DFF, DGS2, DGS10, VIXCLS, DTWEXBGS) | No |
| On-chain (CoinMetrics Community, catalog-discovered) | No — but only whatever the catalog says is actually free (SOPR/NUPL/exchange-flows usually aren't) |
| News (CryptoCompare) | Yes — `CC_API_KEY` in `.env` |
| LLM event classification | Yes — `DEEPSEEK_API_KEY` preferred (cheap), or `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` |

## Leakage tests

```bash
pytest tests/test_leakage.py -v
```

Covers: the target definition itself, `select_prediction_times` landing on
real clock hours (not array indices), the train/validation/test split
being clean under every STEP_HOURS x split combination, the runtime guards
(`assert_no_label_overlap`, `assert_point_in_time`) correctly rejecting
deliberately-corrupted inputs, and `build_master`'s per-source
`*_available_at` naming not colliding across merges.

## Repo layout

```
config.py              single source of truth: dates, target, split, FRED/on-chain lists
run_pipeline.py         orchestrates everything below, end to end
src/
  common.py             UTC helpers, parquet cache, coverage_report
  binance_spot.py        Spot 1h klines (data-api.binance.vision, fallback api.binance.com)
  binance_futures.py     funding rate (REST) + daily metrics (data.binance.vision archive)
  technical_features.py  stationary technicals + spot order flow + time-of-day
  fear_greed.py           Alternative.me F&G, available_at = source_time + 1 day
  news.py                 CryptoCompare collector, resumable checkpoint
  news_features.py        FinBERT sentiment + DeepSeek/Anthropic/OpenAI event classification
  onchain.py              CoinMetrics catalog discovery + generic fetch
  macro.py                FRED CSV endpoint, one frame per series
  master_table.py         asof_merge_source, build_master, report_modalities
  samples.py              create_target, select_prediction_times, split_samples
  leakage.py              assert_no_label_overlap, assert_point_in_time
  models/                 xgb_model.py, tcn.py, cnn_lstm.py, attention_model.py — not wired up yet, see below
  evaluation/             metrics.py
  features/feature_selection.py
data/          raw/ processed/ cache/  (all gitignored)
notebooks/     00_semester1_original (reference only), 01-03 placeholders
```

## Order of work

Don't build models yet. Run the pipeline, read `results/data_coverage.csv`,
and answer these five questions first:

1. Which futures history actually downloaded (archive coverage vs. the
   ~30-day REST-only fallback)?
2. Which CoinMetrics on-chain metrics were genuinely free (see the printed
   catalog match and `data/cache/onchain_discovered_metrics.txt`)?
3. Did CryptoCompare news actually reach back to 2019?
4. What's the common coverage window across every source?
5. At STEP=6, how many samples do we really have, and is the class
   balance still close to 50/50?

Only once those are answered does it make sense to write XGBoost/TCN/
CNN-LSTM training code against this data — the scaffolding for those
already exists in `src/models/` and `src/evaluation/`, deliberately not
wired into `run_pipeline.py` yet.

## Provenance

Semester 1 exploratory notebook is kept as `notebooks/00_semester1_original.ipynb`
for reference — it established the target definition, FinBERT usage, and
leakage-aware windowing this repo generalizes. It mixed hourly Binance data
with daily F&G, weekly Trends, and news via forward/backward fills, and
hardcoded a 2025-2026 date range into the CryptoCompare collector; this
rebuild replaces that with the `available_at` / `merge_asof(backward)`
design above, driven entirely by `config.py`.

**Security note:** that notebook had a live CryptoCompare API key pasted
directly into a code cell. The copy in `notebooks/00_semester1_original.ipynb`
has that one string redacted so the secret never enters git history —
everything else is untouched. Treat the original key as compromised and
never paste a real key into a notebook or commit one again — `.env` only
(git-ignored).
