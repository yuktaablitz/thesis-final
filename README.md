# BTC Multimodal Thesis

Predicts the direction of Binance BTCUSDT spot close, exactly 24 hours ahead,
using two frozen final approaches. No new directions get added unless
ablation/validation results say we need one.

```
Input:  everything known at or before t
Target: is close(t + 24h) > close(t) ?
```

## Approaches

- **Approach 1 — BTC Market Intelligence Model** (`configs/approach1.yaml`):
  spot technicals + order flow, derivatives (funding/OI/basis), on-chain,
  Fear & Greed, FinBERT news sentiment, LLM news-event classification, a
  small macro set. Models: XGBoost/LightGBM, TCN, CNN-LSTM, optional
  ensemble. Build this first.
- **Approach 2 — Cross-Market AI Attention Model** (`configs/approach2.yaml`):
  same BTC target, adds ETH/BNB/SOL as context via separate encoder branches,
  temporal + modality attention. Only build after Approach 1 is working —
  see the decision point below.

Both share one leakage-safe target/split definition (`configs/approach1.yaml`
is the source of truth; `approach2.yaml` extends it).

## Decision point (after Approach 1 ablation)

- Approach 1 test accuracy ~75–80%+ cleanly → Approach 2 becomes an advanced
  thesis experiment, not a rescue attempt.
- Approach 1 stuck at ~60–68% → Approach 2 tests the hypothesis that BTC
  needs broader cross-market context and dynamic attention.

## Leakage-safe split (non-negotiable)

```
TRAIN       2019-01-01 → 2023-12-31
VALIDATION  2024-01-01 → 2024-12-31
TEST        2025-01-01 → latest common usable date
```

```python
train.label_end_time      < validation_start
validation.label_end_time < test_start
```

`tests/test_leakage.py` asserts this automatically — run it after every
change to the sampling code, before trusting any result.

## Setup (local — VS Code, M4 Pro)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name btc-thesis --display-name "BTC Thesis"
cp .env.example .env   # then fill in whichever keys you have
```

In VS Code: install the Python + Jupyter extensions, then select the
`.venv` interpreter / "BTC Thesis" kernel.

- **Local (VS Code + `.venv`, MPS acceleration on Apple Silicon):** data
  download/caching, feature engineering, XGBoost/LightGBM, small experiments.
- **Google Colab (GPU/CUDA):** FinBERT batch inference, TCN/CNN-LSTM/attention
  training. `notebooks/` is meant to be opened there directly.
- Never commit raw data, checkpoints, or `.env` — see `.gitignore`.

## What needs an API key vs. what doesn't

| Source | Key needed? |
|---|---|
| Binance spot/futures (klines, funding, OI, premium index) | No — public REST |
| Fear & Greed (alternative.me) | No |
| Macro (yfinance: DXY, VIX, NASDAQ, S&P, yields) | No |
| On-chain (CoinMetrics Community / Glassnode / CryptoQuant) | CoinMetrics Community: partially free; others: yes (`.env`) |
| News (CryptoCompare) | Yes (`.env`, `CC_API_KEY`) |
| LLM event classification | Yes (`.env`, `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) |

Everything in the "No" rows is already implemented and runnable today.
Everything in the "Yes" rows has a working module with the interface wired
up (caching, point-in-time merge, config) that starts producing data the
moment a key is dropped into `.env` — see `src/data/onchain.py` and
`src/data/news.py`.

## Repo layout

```
src/
  data/          one file per raw source, all with on-disk caching + resume
  features/      raw source -> leakage-safe feature columns
  datasets/      labels.py (target), sequences.py (windows+split), master_table.py (merge_asof)
  models/        xgb_model.py, tcn.py, cnn_lstm.py, attention_model.py
  evaluation/    metrics.py, leakage_checks.py, ablation.py
  utils/         config.py, cache.py, logging.py
configs/         approach1.yaml (source of truth), approach2.yaml (extends it)
notebooks/       01_data_audit, 02_approach1, 03_approach2
tests/           test_leakage.py — run this before trusting any result
```

## Order of work (do not build both approaches at once)

1. **Infrastructure** (this commit): repo, venv, cache, config, downloaders,
   sample generator, leakage tests. Serves both approaches.
2. **Approach 1**: XGBoost → TCN → CNN-LSTM → ablation (Stage 14).
3. **Decision point** (see above).
4. **Approach 2**, only if warranted.

## Provenance

Semester 1 exploratory notebook is kept as `notebooks/00_semester1_original.ipynb`
for reference (logic untouched) — it established: yfinance OHLCV,
RSI/MACD/Bollinger/MA features, leakage-aware windowing, LR/XGBoost/LSTM/BiLSTM
baselines, GDELT + Google Trends + VADER + FinBERT fusion models. This repo
generalizes that into the frozen two-approach design above — same target
definition, now on Binance data with a proper leakage-safe split, caching,
and a modular pipeline instead of one long notebook.

**Security note:** that notebook had a live CryptoCompare API key pasted
directly into a code cell. The copy in `notebooks/00_semester1_original.ipynb`
has that one string redacted so the secret itself never enters git history —
everything else is untouched. Treat the original key as compromised —
rotate/regenerate it at https://www.cryptocompare.com/cryptopian/api-keys
before putting the new one in `.env`. It must never be pasted into a
notebook or committed again.
