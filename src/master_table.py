"""
Binance Spot is the master clock — every other source attaches to it via
point-in-time backward merge_asof on that source's own `available_at`
column, never on `source_time` directly and never with bfill.

Each source's `available_at` is renamed to `<name>_available_at` before
merging (funding_available_at, futures_available_at, fng_available_at,
onchain_available_at, macro_<series>_available_at, ...) specifically so
src/leakage.py's assert_point_in_time can audit every modality
independently — a single shared `available_at` column would get silently
overwritten/suffixed across multiple merges.
"""
import numpy as np
import pandas as pd

from config import DATA_END, DATA_START


def filter_global_range(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    start, end = pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)
    return df[(df[time_col] >= start) & (df[time_col] < end)].copy()


def asof_merge_source(master: pd.DataFrame, source_df: pd.DataFrame, name: str,
                       available_col: str = "available_at") -> pd.DataFrame:
    """
    Backward point-in-time merge: for each master.timestamp, attach the
    newest `source_df` row whose `<name>_available_at <= timestamp`.
    source_df must already have an `available_col` column (default
    "available_at"); it gets renamed to f"{name}_available_at" here.
    """
    if source_df is None or source_df.empty:
        print(f"[master_table] skipping '{name}' — empty/unavailable")
        return master

    renamed_col = f"{name}_available_at"
    right = source_df.rename(columns={available_col: renamed_col}).sort_values(renamed_col)
    left = master.sort_values("timestamp")

    return pd.merge_asof(left, right, left_on="timestamp", right_on=renamed_col, direction="backward")


def build_master(spot: pd.DataFrame, funding: pd.DataFrame | None = None,
                  futures_metrics: pd.DataFrame | None = None, fng: pd.DataFrame | None = None,
                  news_hourly: pd.DataFrame | None = None, onchain: pd.DataFrame | None = None,
                  macro_frames: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    master = spot.sort_values("timestamp").reset_index(drop=True).copy()

    # ---- funding: available the instant it's realized (it IS the funding event) ----
    if funding is not None and not funding.empty:
        f = funding.copy()
        f["available_at"] = f["timestamp"]
        f = f.rename(columns={"timestamp": "funding_source_time"})
        master = asof_merge_source(master, f, "funding")

    # ---- futures daily metrics: a completed daily archive file, usable next day ----
    if futures_metrics is not None and not futures_metrics.empty:
        fm = futures_metrics.rename(columns={"timestamp": "futures_source_time"}).copy()
        fm["available_at"] = fm["futures_source_time"] + pd.Timedelta(days=1)
        master = asof_merge_source(master, fm, "futures")

    # ---- Fear & Greed (already carries source_time/available_at) ----
    if fng is not None and not fng.empty:
        fng = fng.rename(columns={"source_time": "fng_source_time"})
        master = asof_merge_source(master, fng, "fng")

    # ---- news: hourly-bucketed, equality merge (see src/news_features.py docstring) ----
    if news_hourly is not None and not news_hourly.empty:
        master = master.merge(news_hourly, on="timestamp", how="left")
        news_cols = [c for c in news_hourly.columns if c != "timestamp"]
        master[news_cols] = master[news_cols].fillna(0)

    # ---- on-chain (already carries source_time/available_at) ----
    if onchain is not None and not onchain.empty:
        onchain = onchain.rename(columns={"source_time": "onchain_source_time"})
        master = asof_merge_source(master, onchain, "onchain")

    # ---- macro: one FRED series at a time, each with its own available_at ----
    if macro_frames:
        for frame in macro_frames:
            if frame is None or frame.empty:
                continue
            series_name = [c for c in frame.columns if c not in ("source_time", "available_at")][0]
            renamed = frame.rename(columns={"source_time": f"{series_name}_source_time"})
            master = asof_merge_source(master, renamed, series_name)

    return master.replace([np.inf, -np.inf], np.nan)


def add_missing_indicators(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Adds `<col>_missing` flags instead of dropping rows — this is how Jan-Aug
    2019 (spot exists, derivatives don't; Binance futures launched
    2019-09-13) stays in the table rather than being deleted. XGBoost/
    LightGBM can often use the raw NaN directly; impute for neural nets
    AFTER splitting (fit imputers on train only).
    """
    df = df.copy()
    for col in feature_cols:
        if col in df.columns and df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype("int8")
    return df


def report_modalities(df: pd.DataFrame) -> pd.DataFrame:
    """Prints and returns per-modality coverage — run this BEFORE modeling, save to
    results/data_coverage.csv. Fails loudly (via the printed ranges) if a source's
    dates don't make sense, e.g. derivatives coverage starting in 2019 instead of Sep 2019."""
    groups = {
        "spot": ["close", "volume", "spot_taker_imbalance"],
        "derivatives": ["funding_rate", "futures_open_interest"],
        "sentiment": ["fear_greed"],
        "news": ["news_count", "news_sentiment"],
        "onchain": [c for c in df.columns if c.startswith("Cap") or c.startswith("Adr") or c.startswith("Tx")],
    }

    rows = []
    for name, cols in groups.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            print(f"{name:12s} NO COLUMNS")
            rows.append({"modality": name, "coverage_start": None, "coverage_end": None, "row_fraction": 0.0})
            continue

        mask = df[available].notna().any(axis=1)
        if mask.any():
            start, end, frac = df.loc[mask, "timestamp"].min(), df.loc[mask, "timestamp"].max(), mask.mean()
            print(f"{name:12s} {start} -> {end} | {frac:.1%} of rows")
            rows.append({"modality": name, "coverage_start": start, "coverage_end": end, "row_fraction": frac})
        else:
            print(f"{name:12s} no non-null rows")
            rows.append({"modality": name, "coverage_start": None, "coverage_end": None, "row_fraction": 0.0})

    return pd.DataFrame(rows)
