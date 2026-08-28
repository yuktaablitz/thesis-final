"""Stage 12A — XGBoost/LightGBM tabular baseline. Consumes flattened windows."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """(n_samples, window_hours, n_features) -> (n_samples, window_hours * n_features)."""
    return X.reshape(X.shape[0], -1)


def make_xgb(**overrides) -> XGBClassifier:
    params = dict(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    params.update(overrides)
    return XGBClassifier(**params)


def make_lgbm(**overrides) -> LGBMClassifier:
    params = dict(
        n_estimators=400, max_depth=-1, num_leaves=31, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        objective="binary", random_state=42, n_jobs=-1, verbosity=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)
