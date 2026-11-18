"""
Automated leakage tests for the rebuilt pipeline (config.py + src/samples.py
+ src/leakage.py). Run after any change to those modules, before trusting
any result:

    pytest tests/test_leakage.py -v
"""
import numpy as np
import pandas as pd
import pytest

from config import DATA_END, DATA_START, TEST_START, VALIDATION_START
from src.leakage import assert_no_label_overlap, assert_point_in_time
from src.samples import create_target, select_prediction_times, split_samples


@pytest.fixture
def synthetic_master() -> pd.DataFrame:
    """Synthetic hourly BTC-like data spanning config.DATA_START -> DATA_END, no network calls."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(DATA_START, DATA_END, freq="1h", tz="UTC", inclusive="left")
    log_returns = rng.normal(0, 0.003, size=len(timestamps))
    close = 5000 * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
        "volume": rng.uniform(100, 1000, size=len(timestamps)),
    })
    return create_target(df)


def test_target_definition(synthetic_master):
    df = synthetic_master
    row, future_row = df.iloc[1000], df.iloc[1024]
    assert future_row["timestamp"] - row["timestamp"] == pd.Timedelta(hours=24)
    expected_return = future_row["close"] / row["close"] - 1
    assert np.isclose(row["future_return_24h"], expected_return)
    assert int(row["target"]) == int(expected_return > 0)
    assert row["label_end_time"] == row["timestamp"] + pd.Timedelta(hours=24)


def test_last_rows_unlabeled(synthetic_master):
    tail = synthetic_master.tail(24)
    assert tail["target"].isna().all()


@pytest.mark.parametrize("step_hours,expected_hours", [(6, {0, 6, 12, 18}), (24, {0}), (1, set(range(24)))])
def test_select_prediction_times_uses_clock_hours(synthetic_master, step_hours, expected_hours):
    samples = select_prediction_times(synthetic_master, step_hours=step_hours)
    assert set(samples["timestamp"].dt.hour.unique()) == expected_hours


@pytest.mark.parametrize("step_hours", [1, 6, 24])
def test_split_is_leakage_safe(synthetic_master, step_hours):
    samples = select_prediction_times(synthetic_master, step_hours=step_hours)
    train, val, test = split_samples(samples)

    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    val_start, test_start = pd.Timestamp(VALIDATION_START), pd.Timestamp(TEST_START)

    assert train["label_end_time"].max() < val_start
    assert val["timestamp"].min() >= val_start
    assert val["label_end_time"].max() < test_start
    assert test["timestamp"].min() >= test_start

    # assert_no_label_overlap must accept this split without raising.
    assert_no_label_overlap(train, val, test)


def test_leakage_guard_catches_corrupted_splits():
    """Independent test of the runtime guard: feed it a manually-built, deliberately leaky
    set of splits (bypassing split_samples' own filtering) and confirm it raises."""
    val_start = pd.Timestamp(VALIDATION_START)

    leaky_train = pd.DataFrame({
        "timestamp": [val_start - pd.Timedelta(hours=2), val_start - pd.Timedelta(hours=1)],
        "label_end_time": [val_start + pd.Timedelta(hours=40), val_start + pd.Timedelta(hours=41)],
    })
    validation = pd.DataFrame({
        "timestamp": [val_start], "label_end_time": [val_start + pd.Timedelta(hours=24)],
    })
    test = pd.DataFrame({"timestamp": [], "label_end_time": []})

    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_label_overlap(leaky_train, validation, test)


def test_leakage_guard_catches_duplicate_timestamps():
    ts = pd.Timestamp(DATA_START)
    dup = pd.DataFrame({"timestamp": [ts, ts], "label_end_time": [ts + pd.Timedelta(hours=24)] * 2})
    empty = pd.DataFrame({"timestamp": [], "label_end_time": []})
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_label_overlap(dup, empty, empty)


def test_assert_point_in_time_passes_when_clean():
    ts = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "fng_available_at": ts - pd.Timedelta(hours=1),  # known before prediction time: fine
    })
    assert_point_in_time(df)  # must not raise


def test_assert_point_in_time_catches_future_leak():
    ts = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "onchain_available_at": ts + pd.Timedelta(hours=1),  # known AFTER prediction time: a leak
    })
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_point_in_time(df)
