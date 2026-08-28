"""Target definition, sample-time selection, and the chronological train/validation/test split."""
import pandas as pd

from config import HORIZON_HOURS, TEST_END, TEST_START, TRAIN_START, VALIDATION_START


def create_target(df: pd.DataFrame, horizon_hours: int = HORIZON_HOURS) -> pd.DataFrame:
    """
    Input:  everything known at or before t (timestamp)
    Target: is close(t + horizon_hours) > close(t) ?
    Preserves the final `horizon_hours` rows as NaN (pd.NA) targets rather
    than silently coercing them to 0 — a bug in the old EDA notebook.
    """
    df = df.sort_values("timestamp").copy()
    h = horizon_hours

    df["current_close"] = df["close"]
    df["future_close_24h"] = df["close"].shift(-h)
    df["future_return_24h"] = df["future_close_24h"] / df["current_close"] - 1

    valid = df["future_close_24h"].notna()
    df["target"] = pd.NA
    df.loc[valid, "target"] = (df.loc[valid, "future_close_24h"] > df.loc[valid, "current_close"]).astype(int)

    df["label_end_time"] = df["timestamp"] + pd.Timedelta(hours=h)
    return df


def select_prediction_times(df: pd.DataFrame, step_hours: int) -> pd.DataFrame:
    """Selects rows on actual clock-hour boundaries (00/06/12/18 for step=6), not by array index."""
    mask = df["timestamp"].dt.hour % step_hours == 0
    return df.loc[mask].reset_index(drop=True)


def split_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits by LABEL END TIME, not just prediction_time — this is the actual
    leakage protection:
      train:      label_end_time      < validation_start
      validation: label_end_time      < test_start (and timestamp >= validation_start)
      test:       label_end_time      < test_end   (and timestamp >= test_start)
    """
    df = df.dropna(subset=["target"])
    train_start = pd.Timestamp(TRAIN_START)
    val_start = pd.Timestamp(VALIDATION_START)
    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)

    train = df[(df["timestamp"] >= train_start) & (df["label_end_time"] < val_start)].copy()
    validation = df[(df["timestamp"] >= val_start) & (df["label_end_time"] < test_start)].copy()
    test = df[(df["timestamp"] >= test_start) & (df["label_end_time"] < test_end)].copy()

    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
