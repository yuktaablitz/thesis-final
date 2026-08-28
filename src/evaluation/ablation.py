"""
Stage 14 — ablation. Runs the cumulative feature-family sweep from
configs/approach1.yaml `ablation_order` (technical-only, + spot order flow,
+ derivatives, ... , FULL), training the same model family at each step so
the marginal contribution of each source is comparable.

This is intentionally model-agnostic: pass in any `train_fn` with the
signature (X_train, y_train, X_val, y_val) -> y_val_prob, and any
`select_columns_fn(df, enabled_families) -> list[str]` that maps the
config's family flags to concrete column names for your feature set.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate, results_table
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger("evaluation.ablation")


def run_ablation(
    train_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    build_xy_fn: Callable[[list[str]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """
    train_fn(X_train, y_train, X_val, y_val) -> y_val_prob
    build_xy_fn(enabled_families) -> (X_train, y_train, X_val, y_val), already
        filtered to only the columns implied by `enabled_families`.

    Runs one step per prefix of configs/approach1.yaml `ablation_order`
    (cumulative — "technical only", "+ order flow", "+ derivatives", ...).
    """
    order = load_config("approach1.yaml")["ablation_order"]
    results = []
    enabled: list[str] = []

    for family in order:
        enabled = enabled + [family]
        step_name = " + ".join(enabled) if len(enabled) > 1 else enabled[0]
        log.info(f"ablation step: {step_name}")

        X_train, y_train, X_val, y_val = build_xy_fn(enabled)
        y_prob = train_fn(X_train, y_train, X_val, y_val)
        metrics = evaluate(step_name, y_val, y_prob, verbose=False)
        results.append(metrics)

    table = results_table(results)
    log.info(f"\n{table}")
    return table
