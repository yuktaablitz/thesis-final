"""
Stage 10 — sample generation. Builds the sample index (window_start,
prediction_time, label_end_time) at a given STEP_HOURS cadence, over a
given window length, and enforces the leakage-safe chronological split
from configs/approach1.yaml. This is the one place that decides which
hours become samples — src.models.* consumes its output, never the raw
hourly frame directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger("datasets.sequences")


def assert_no_gaps(hourly_df: pd.DataFrame, max_gap_hours: int = 1) -> None:
    """Warn (don't crash) on missing hours — exchange downtime happens; big gaps shouldn't."""
    diffs = hourly_df["timestamp"].diff().dropna()
    expected = pd.Timedelta(hours=max_gap_hours)
    bad = diffs[diffs > expected]
    if len(bad):
        log.warning(f"{len(bad)} gap(s) > {max_gap_hours}h in the hourly timeline; "
                    f"largest = {bad.max()}")


def build_sample_index(hourly_df: pd.DataFrame, window_hours: int, step_hours: int,
                        horizon_hours: int | None = None) -> pd.DataFrame:
    """
    hourly_df: output of src.datasets.labels.add_target (has timestamp, target,
    label_end_time, sorted ascending, one row/hour).

    Returns one row per sample: window_start, prediction_time, label_end_time,
    target, row_idx (position of prediction_time in hourly_df, so callers can
    slice hourly_df.iloc[row_idx - window_hours : row_idx] for the feature window).
    """
    cfg = load_config("approach1.yaml")
    horizon_hours = horizon_hours or cfg["target"]["horizon_hours"]

    df = hourly_df.reset_index(drop=True)
    n = len(df)
    rows = []
    # i is the position of prediction_time; need `window_hours` of history before it
    # and a valid (non-NaN) target, i.e. i must be within range for the horizon shift.
    for i in range(window_hours, n - horizon_hours + 1, step_hours):
        if pd.isna(df.loc[i, "target"]):
            continue
        rows.append({
            "row_idx": i,
            "window_start": df.loc[i - window_hours, "timestamp"],
            "prediction_time": df.loc[i, "timestamp"],
            "label_end_time": df.loc[i, "label_end_time"],
            "target": int(df.loc[i, "target"]),
        })

    idx = pd.DataFrame(rows)
    log.info(f"built {len(idx)} samples (window={window_hours}h, step={step_hours}h)")
    return idx


def chronological_split(sample_index: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Splits a sample index into train/validation/test using configs/approach1.yaml
    `split`, then asserts the leakage-safe condition:
        train.label_end_time      < validation_start
        validation.label_end_time < test_start
    Raises AssertionError if violated — never silently returns a leaky split.
    """
    cfg = load_config("approach1.yaml")["split"]
    train_start = pd.Timestamp(cfg["train_start"], tz="UTC")
    val_start = pd.Timestamp(cfg["validation_start"], tz="UTC")
    test_start = pd.Timestamp(cfg["test_start"], tz="UTC")
    test_end = pd.Timestamp(cfg["test_end"], tz="UTC") if cfg["test_end"] else None

    train = sample_index[
        (sample_index["prediction_time"] >= train_start) & (sample_index["label_end_time"] < val_start)
    ]
    validation = sample_index[
        (sample_index["prediction_time"] >= val_start) & (sample_index["label_end_time"] < test_start)
    ]
    test = sample_index[sample_index["prediction_time"] >= test_start]
    if test_end is not None:
        test = test[test["prediction_time"] <= test_end]

    if len(train):
        assert train["label_end_time"].max() < val_start, (
            f"LEAKAGE: train.label_end_time.max()={train['label_end_time'].max()} "
            f">= validation_start={val_start}"
        )
    if len(validation):
        assert validation["label_end_time"].max() < test_start, (
            f"LEAKAGE: validation.label_end_time.max()={validation['label_end_time'].max()} "
            f">= test_start={test_start}"
        )

    log.info(f"split -> train={len(train)}, validation={len(validation)}, test={len(test)}")
    return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def build_windows(hourly_df: pd.DataFrame, feature_cols: list[str], sample_index: pd.DataFrame,
                   window_hours: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Materializes X (n_samples, window_hours, n_features) and y (n_samples,)
    for sequence models (TCN/CNN-LSTM/attention). For tabular models
    (XGBoost), flatten X or engineer summary stats over the window instead.
    """
    df = hourly_df.reset_index(drop=True)
    X = np.stack([
        df.loc[row_idx - window_hours:row_idx - 1, feature_cols].to_numpy()
        for row_idx in sample_index["row_idx"]
    ])
    y = sample_index["target"].to_numpy()
    return X, y
