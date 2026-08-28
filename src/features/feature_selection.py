"""
Stage 11 — feature selection. Fit everything on TRAIN (+ evaluate on
VALIDATION) only — never touch TEST here, or every downstream result is
contaminated by feature-selection leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("features.feature_selection")


def drop_correlated(df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.95) -> list[str]:
    """Greedy: for each pair above `threshold` |corr|, drop the second-seen column."""
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [c for c in feature_cols if c not in to_drop]
    log.info(f"correlation filter (>{threshold}): dropped {len(to_drop)}, kept {len(kept)}")
    return kept


def boruta_select(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> list[str]:
    try:
        from boruta import BorutaPy
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        log.warning("boruta not installed (`pip install boruta`) — skipping this step")
        return list(X_train.columns)

    rf = RandomForestClassifier(n_jobs=-1, max_depth=7, random_state=random_state)
    selector = BorutaPy(rf, n_estimators="auto", random_state=random_state, verbose=0)
    selector.fit(X_train.values, y_train.values)
    selected = list(X_train.columns[selector.support_])
    log.info(f"Boruta: {len(selected)}/{X_train.shape[1]} confirmed relevant")
    return selected


def xgb_importance(X_train: pd.DataFrame, y_train: pd.Series, top_n: int | None = None) -> pd.Series:
    from xgboost import XGBClassifier

    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return importances.head(top_n) if top_n else importances


def shap_values(model, X: pd.DataFrame, max_samples: int = 2000):
    """Returns a shap.Explanation for interpretation (Stage 11 step 4) — plot in a notebook, not here."""
    import shap

    sample = X.sample(min(max_samples, len(X)), random_state=42) if len(X) > max_samples else X
    explainer = shap.TreeExplainer(model)
    return explainer(sample)
