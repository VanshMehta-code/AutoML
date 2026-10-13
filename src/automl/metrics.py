"""The single scoring function used everywhere: AUC for classification, R² for regression.

Higher is always better, so search, ensembling and reporting can share one code path.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score


def task_score(y_true: np.ndarray, pred: np.ndarray, task: str, n_classes: int) -> float:
    if task == "regression":
        return float(r2_score(y_true, pred))
    if n_classes == 2:
        return float(roc_auc_score(y_true, pred[:, 1]))
    return float(roc_auc_score(y_true, pred, multi_class="ovr", labels=np.arange(n_classes)))
