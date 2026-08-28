"""
Runtime (not just test-time) leakage guard — call this right before
training/evaluating any model, on whatever split dict you actually built,
so a leaky split can never silently reach a model even outside pytest.
"""
from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

log = get_logger("evaluation.leakage_checks")


def assert_leakage_safe_splits(splits: dict[str, pd.DataFrame]) -> None:
    train, validation, test = splits["train"], splits["validation"], splits["test"]

    if len(train) and len(validation):
        assert train["label_end_time"].max() < validation["prediction_time"].min(), (
            "LEAKAGE: train labels extend into the validation prediction window"
        )
    if len(validation) and len(test):
        assert validation["label_end_time"].max() < test["prediction_time"].min(), (
            "LEAKAGE: validation labels extend into the test prediction window"
        )

    for split_name, split_df in splits.items():
        if len(split_df) and split_df["row_idx"].duplicated().any():
            raise AssertionError(f"LEAKAGE: duplicate row_idx within '{split_name}' split")

    row_idx_sets = {name: set(df["row_idx"]) for name, df in splits.items() if len(df)}
    names = list(row_idx_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = row_idx_sets[names[i]] & row_idx_sets[names[j]]
            assert not overlap, f"LEAKAGE: {names[i]} and {names[j]} share {len(overlap)} sample(s)"

    log.info("leakage checks passed: "
              f"train={len(train)}, validation={len(validation)}, test={len(test)}")
