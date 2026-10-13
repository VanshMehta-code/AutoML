"""Leakage screening: each feature is scored alone against the target, on training rows only.

A feature that predicts the target almost perfectly by itself is usually a leak:
it was computed from the outcome, or it is the outcome under another name.
This check is statistical. It cannot catch a leak that is weak on its own, and it
cannot tell whether a strong feature is legitimately known before prediction time.
Those judgements still need a person and the dataset documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .metadata import CONSTANT, DATETIME, IDENTIFIER, NUMERIC
from .metrics import task_score


@dataclass
class LeakReport:
    scores: dict[str, float] = field(default_factory=dict)
    flagged: list[str] = field(default_factory=list)
    suspicious: list[str] = field(default_factory=list)


def _as_numeric(s: pd.Series, kind: str) -> np.ndarray:
    if kind == NUMERIC:
        x = pd.to_numeric(s, errors="coerce").astype(float)
    elif kind == DATETIME:
        d = pd.to_datetime(s, errors="coerce", format="mixed")
        x = (d - pd.Timestamp("1970-01-01")).dt.total_seconds()
    else:
        return pd.factorize(s.where(s.notna(), "__missing__").astype(str))[0].astype(float)
    fill = x.median() if x.notna().any() else 0.0
    return x.fillna(fill).to_numpy()


def screen_leakage(
    X: pd.DataFrame,
    y: np.ndarray,
    meta: dict,
    task: str,
    seed: int = 0,
    threshold: float = 0.95,
    warn_threshold: float = 0.85,
) -> LeakReport:
    """Score every usable feature with a depth-3 tree trained on that feature alone."""
    report = LeakReport()
    is_reg = task == "regression"
    n_classes = meta.get("n_classes") or 0

    if is_reg:
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        model, method = DecisionTreeRegressor(max_depth=3, random_state=seed), "predict"
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        model, method = DecisionTreeClassifier(max_depth=3, random_state=seed), "predict_proba"

    for col, info in meta["columns"].items():
        if info["kind"] in (CONSTANT, IDENTIFIER):
            continue
        x = _as_numeric(X[col], info["kind"]).reshape(-1, 1)
        try:
            pred = cross_val_predict(model, x, y, cv=cv, method=method)
            score = task_score(y, pred, task, n_classes)
        except ValueError:  # e.g. too few rows of some class for stratified folds
            continue
        report.scores[col] = score
        if score >= threshold:
            report.flagged.append(col)
        elif score >= warn_threshold:
            report.suspicious.append(col)
    return report
