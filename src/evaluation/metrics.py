"""Shared evaluation — every model (XGBoost, TCN, CNN-LSTM, attention) reports through this."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score,
)

from src.utils.logging import get_logger

log = get_logger("evaluation.metrics")


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
             verbose: bool = True) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "name": name,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    if verbose:
        log.info(f"========== {name} ==========")
        log.info(f"Accuracy : {metrics['accuracy']:.4f}")
        log.info(f"F1       : {metrics['f1']:.4f}")
        log.info(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        log.info(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")
        log.info(f"\n{classification_report(y_true, y_pred, zero_division=0)}")
    return metrics


def results_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).set_index("name").sort_values("roc_auc", ascending=False)
