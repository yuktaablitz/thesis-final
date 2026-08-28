"""
Automated leakage tests. Run these after any change to
src/datasets/{labels,sequences}.py — before trusting any experiment result.

    pytest tests/test_leakage.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datasets.labels import add_target
from src.datasets.sequences import build_sample_index, chronological_split
from src.evaluation.leakage_checks import assert_leakage_safe_splits


@pytest.fixture
def synthetic_hourly_df() -> pd.DataFrame:
    """~6.5 years of synthetic hourly BTC-like data, no network calls."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2019-01-01", "2025-06-01", freq="1h", tz="UTC")
    log_returns = rng.normal(0, 0.003, size=len(timestamps))
    close = 5000 * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
        "volume": rng.uniform(100, 1000, size=len(timestamps)),
    })
    return add_target(df)


def test_target_definition(synthetic_hourly_df):
    df = synthetic_hourly_df
    row = df.iloc[100]
    future_row = df.iloc[100 + 24]
    assert future_row["timestamp"] - row["timestamp"] == pd.Timedelta(hours=24)
    expected_return = future_row["close"] / row["close"] - 1
    assert np.isclose(row["future_return_24h"], expected_return)
    assert row["target"] == int(expected_return > 0)
    assert row["label_end_time"] == row["timestamp"] + pd.Timedelta(hours=24)


def test_last_rows_unlabeled(synthetic_hourly_df):
    tail = synthetic_hourly_df.tail(24)
    assert tail["target"].isna().all()


@pytest.mark.parametrize("step_hours", [1, 6, 24])
@pytest.mark.parametrize("window_hours", [72, 168])
def test_split_is_leakage_safe(synthetic_hourly_df, window_hours, step_hours):
    """The condition this whole test file exists to enforce:
        train.label_end_time      < validation_start
        validation.label_end_time < test_start
    """
    idx = build_sample_index(synthetic_hourly_df, window_hours=window_hours, step_hours=step_hours)
    splits = chronological_split(idx)  # raises AssertionError internally if violated

    train, validation, test = splits["train"], splits["validation"], splits["test"]
    assert len(train) > 0 and len(validation) > 0 and len(test) > 0

    val_start = pd.Timestamp("2024-01-01", tz="UTC")
    test_start = pd.Timestamp("2025-01-01", tz="UTC")
    assert train["label_end_time"].max() < val_start
    assert validation["prediction_time"].min() >= val_start
    assert validation["label_end_time"].max() < test_start
    assert test["prediction_time"].min() >= test_start

    # No sample appears in two splits.
    train_idx, val_idx, test_idx = set(train["row_idx"]), set(validation["row_idx"]), set(test["row_idx"])
    assert not (train_idx & val_idx)
    assert not (val_idx & test_idx)
    assert not (train_idx & test_idx)


def test_chronological_split_excludes_boundary_crossers(synthetic_hourly_df):
    """
    chronological_split's train/validation filters ARE the leakage condition
    (label_end_time < next period's start) — so a sample whose label would
    cross into the next period is dropped from the earlier split rather than
    included-then-flagged. Confirm that drop actually happens at the
    train/validation boundary (Dec 31 23:00 prediction_time -> label_end_time
    Jan 1 23:00, which is >= validation_start and must NOT end up in train).
    """
    idx = build_sample_index(synthetic_hourly_df, window_hours=72, step_hours=1)
    splits = chronological_split(idx)
    val_start = pd.Timestamp("2024-01-01", tz="UTC")

    boundary_crossers = idx[
        (idx["prediction_time"] < val_start) & (idx["label_end_time"] >= val_start)
    ]
    assert len(boundary_crossers) > 0, "test fixture should include some boundary-crossing samples"
    assert not set(boundary_crossers["row_idx"]) & set(splits["train"]["row_idx"]), (
        "a sample whose label reaches into the validation period leaked into train"
    )


def test_leakage_guard_catches_corrupted_splits():
    """
    Independent test of the runtime guard (src.evaluation.leakage_checks),
    used in notebooks right before training on whatever split dict was
    actually built. Feed it a manually-constructed, deliberately leaky
    splits dict (bypassing chronological_split's own filtering entirely)
    and confirm it raises rather than passing silently.
    """
    val_start = pd.Timestamp("2024-01-01", tz="UTC")
    test_start = pd.Timestamp("2025-01-01", tz="UTC")

    leaky_train = pd.DataFrame({
        "row_idx": [1, 2],
        "prediction_time": [val_start - pd.Timedelta(hours=2), val_start - pd.Timedelta(hours=1)],
        # These labels reach 40h past validation_start — a real leak.
        "label_end_time": [val_start + pd.Timedelta(hours=40), val_start + pd.Timedelta(hours=41)],
    })
    validation = pd.DataFrame({
        "row_idx": [3], "prediction_time": [val_start], "label_end_time": [val_start + pd.Timedelta(hours=24)],
    })
    test = pd.DataFrame({
        "row_idx": [4], "prediction_time": [test_start], "label_end_time": [test_start + pd.Timedelta(hours=24)],
    })

    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_leakage_safe_splits({"train": leaky_train, "validation": validation, "test": test})


def test_leakage_guard_catches_overlapping_row_idx():
    """Same sample appearing in two splits must also be caught, independent of timestamps."""
    shared = pd.DataFrame({
        "row_idx": [10], "prediction_time": [pd.Timestamp("2023-06-01", tz="UTC")],
        "label_end_time": [pd.Timestamp("2023-06-02", tz="UTC")],
    })
    empty_like = shared.iloc[0:0]
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_leakage_safe_splits({"train": shared, "validation": shared, "test": empty_like})


def test_window_start_matches_window_hours(synthetic_hourly_df):
    window_hours = 72
    idx = build_sample_index(synthetic_hourly_df, window_hours=window_hours, step_hours=24)
    sample = idx.iloc[10]
    assert sample["prediction_time"] - sample["window_start"] == pd.Timedelta(hours=window_hours)
