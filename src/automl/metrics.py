"""Scoring helpers. The headline metric is ROC AUC: binary directly, multiclass one-vs-rest."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    if n_classes == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=np.arange(n_classes)))
