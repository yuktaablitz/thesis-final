"""
Automated leakage assertions. run_pipeline.py calls both of these and stops
execution (lets the AssertionError propagate) if either fails — a leaky
pipeline should never silently produce a result.
"""
import pandas as pd


def assert_no_label_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    if len(train) and len(val):
        assert train["label_end_time"].max() < val["timestamp"].min(), (
            "LEAKAGE: training labels enter validation!"
        )
    if len(val) and len(test):
        assert val["label_end_time"].max() < test["timestamp"].min(), (
            "LEAKAGE: validation labels enter test!"
        )

    for split_name, split_df in (("train", train), ("validation", val), ("test", test)):
        if len(split_df) and split_df["timestamp"].duplicated().any():
            raise AssertionError(f"LEAKAGE: duplicate timestamp within '{split_name}' split")

    ts_sets = {"train": set(train["timestamp"]), "validation": set(val["timestamp"]), "test": set(test["timestamp"])}
    names = list(ts_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = ts_sets[names[i]] & ts_sets[names[j]]
            assert not overlap, f"LEAKAGE: {names[i]} and {names[j]} share {len(overlap)} timestamp(s)"

    print(f"[OK] label boundaries clean (train={len(train)}, val={len(val)}, test={len(test)})")


def assert_point_in_time(df: pd.DataFrame) -> None:
    """
    Checks every `<name>_available_at` column produced by src/master_table.py:
    a merged value must never have been available AFTER the row's own
    prediction timestamp.
    """
    availability_cols = [c for c in df.columns if c.endswith("_available_at")]
    if not availability_cols:
        print("[WARN] no *_available_at columns found — nothing to audit "
              "(did build_master() actually merge any point-in-time source?)")
        return

    for col in availability_cols:
        valid = df[col].notna()
        if not valid.any():
            continue
        assert (df.loc[valid, col] <= df.loc[valid, "timestamp"]).all(), f"LEAKAGE in {col}"

    print(f"[OK] point-in-time features clean ({len(availability_cols)} sources audited: {availability_cols})")
