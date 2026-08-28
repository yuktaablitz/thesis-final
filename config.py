"""
Single source of truth for dates, target, sampling and split boundaries.
Every downloader/feature/pipeline module imports from here — no module may
hardcode its own experiment date range (that was the Semester 1 bug: the
CryptoCompare collector had START_DATE/END_DATE baked into the cell).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for p in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODEL_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBAL TIME RANGE
# ============================================================

DATA_START = "2019-01-01 00:00:00+00:00"

# Keep this FIXED during thesis experiments. Do not continuously extend the
# test set — every re-run should be reproducing the same frozen window.
DATA_END = "2026-04-01 00:00:00+00:00"

# Binance USD-M futures launched 2019-09-13. Spot/F&G/on-chain start at
# DATA_START; derivatives legitimately start later. Never drop Jan-Aug 2019
# spot rows just because derivatives are missing there — see
# src/master_table.py's missingness handling.
FUTURES_LAUNCH = "2019-09-13 00:00:00+00:00"


# ============================================================
# TARGET
# ============================================================

HORIZON_HOURS = 24

# Primary experiment cadence (predictions at 00/06/12/18 UTC).
STEP_HOURS = 6

# Benchmark (one prediction/day) and sensitivity-only cadences — not the
# headline result, see src/samples.py.
BENCHMARK_STEP_HOURS = 24
SENSITIVITY_STEP_HOURS = 1

# Sequential models (TCN/CNN-LSTM/attention) look back this many hours.
WINDOW_HOURS = 72


# ============================================================
# SPLITS
# ============================================================

TRAIN_START = "2019-01-01 00:00:00+00:00"
VALIDATION_START = "2024-01-01 00:00:00+00:00"
TEST_START = "2025-01-01 00:00:00+00:00"

# TEST ends at DATA_END.
TEST_END = DATA_END


SYMBOL = "BTCUSDT"

# Small, fixed macro set (FRED series IDs — no API key required for the CSV endpoint).
FRED_SERIES = ["DFF", "DGS2", "DGS10", "VIXCLS", "DTWEXBGS"]

# On-chain metric name fragments to keep from CoinMetrics' catalog discovery
# (see src/onchain.py discover_metrics) — kept broad on purpose; discovery
# decides what's actually free, this just filters candidates.
ONCHAIN_KEYWORDS = ["mvrv", "realized", "sopr", "nupl", "active", "transaction", "fee", "hash", "supply", "priceusd"]
