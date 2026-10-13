"""Cross-validated scoring with fidelity control and caching.

The evaluator works on an already-encoded matrix. Preprocessing is fitted once
on the training split, so every candidate scores on the same features, and the
search never re-runs cleaning. Results are cached by a hash of
(family, params, fraction, mask), so duplicate candidates cost nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split

from . import models
from .metrics import task_score


@dataclass
class Evaluation:
    family: str
    params: dict
    score: float            # pooled out-of-fold score on the rows used at this fidelity
    fraction: float         # share of training rows used
    oof: np.ndarray | None  # out-of-fold predictions, only at full fidelity

    def key(self) -> str:
        return json.dumps([self.family, self.params], sort_keys=True, default=str)


class Evaluator:
    def __init__(self, X: np.ndarray, y: np.ndarray, task: str, n_classes: int,
                 cv_folds: int = 5, seed: int = 0):
        self.X = X
        self.y = y
        self.task = task
        self.n_classes = n_classes
        self.cv_folds = cv_folds
        self.seed = seed
        self._cache: dict[str, Evaluation] = {}

    def _rows(self, fraction: float) -> np.ndarray:
        n = len(self.y)
        if fraction >= 1.0:
            return np.arange(n)
        stratify = None if self.task == "regression" else self.y
        try:
            rows, _ = train_test_split(np.arange(n), train_size=fraction, stratify=stratify, random_state=self.seed)
        except ValueError:  # a class too rare to stratify at this fraction
            rows, _ = train_test_split(np.arange(n), train_size=fraction, random_state=self.seed)
        return np.sort(rows)

    def evaluate(self, family: str, params: dict, fraction: float = 1.0,
                 mask: np.ndarray | None = None) -> Evaluation:
        key = _cache_key(family, params, fraction, mask)
        if key in self._cache:
            return self._cache[key]

        rows = self._rows(fraction)
        X = self.X if mask is None else self.X[:, mask]
        X, y = X[rows], self.y[rows]

        if self.task == "regression":
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            method = "predict"
        else:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            method = "predict_proba"

        pred = cross_val_predict(models.build(family, params, self.task), X, y, cv=cv, method=method)
        result = Evaluation(
            family=family,
            params=dict(params),
            score=task_score(y, pred, self.task, self.n_classes),
            fraction=fraction,
            oof=pred if fraction >= 1.0 else None,
        )
        self._cache[key] = result
        return result


def _cache_key(family: str, params: dict, fraction: float, mask: np.ndarray | None) -> str:
    h = hashlib.sha1(json.dumps([family, params, round(fraction, 4)], sort_keys=True, default=str).encode())
    if mask is not None:
        h.update(np.packbits(mask).tobytes())
    return h.hexdigest()
