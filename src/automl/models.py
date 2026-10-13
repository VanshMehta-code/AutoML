"""Model families, their hyperparameter search spaces, and the operators that evolve them.

Search-space entries are (kind, low, high) with kind in {"loguniform", "uniform", "int"}.
Add a family by adding one entry to REGISTRY; the search picks it up automatically.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REGISTRY: dict[str, dict] = {
    "hgb": {
        "cls": HistGradientBoostingClassifier,
        "fixed": {"max_iter": 300, "early_stopping": True, "random_state": 0},
        "space": {
            "learning_rate": ("loguniform", 0.01, 0.3),
            "max_leaf_nodes": ("int", 8, 128),
            "min_samples_leaf": ("int", 5, 100),
            "l2_regularization": ("uniform", 0.0, 5.0),
        },
    },
    "rf": {
        "cls": RandomForestClassifier,
        "fixed": {"n_estimators": 300, "n_jobs": -1, "random_state": 0},
        "space": {
            "max_depth": ("int", 3, 20),
            "min_samples_leaf": ("int", 1, 20),
            "max_features": ("uniform", 0.2, 1.0),
        },
    },
    "logreg": {
        "cls": LogisticRegression,
        "fixed": {"max_iter": 2000},
        "space": {"C": ("loguniform", 1e-3, 1e2)},
    },
}


def build(family: str, params: dict):
    """Instantiate a model. Logistic regression is wrapped with scaling, since it needs it."""
    if family not in REGISTRY:
        raise KeyError(f"Unknown model family '{family}'. Known: {sorted(REGISTRY)}")
    spec = REGISTRY[family]
    est = spec["cls"](**{**spec["fixed"], **params})
    if family == "logreg":
        return make_pipeline(StandardScaler(), est)
    return est


def _draw(kind: str, lo: float, hi: float, rng: np.random.Generator):
    if kind == "loguniform":
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if kind == "uniform":
        return float(rng.uniform(lo, hi))
    if kind == "int":
        return int(rng.integers(lo, hi + 1))
    raise ValueError(f"Unknown search-space kind '{kind}'")


def sample_params(family: str, rng: np.random.Generator) -> dict:
    return {name: _draw(*spec, rng) for name, spec in REGISTRY[family]["space"].items()}


def mutate(family: str, params: dict, rng: np.random.Generator, rate: float = 0.3) -> dict:
    """Resample each hyperparameter with probability `rate`."""
    out = dict(params)
    for name, spec in REGISTRY[family]["space"].items():
        if rng.random() < rate:
            out[name] = _draw(*spec, rng)
    return out


def crossover(a: dict, b: dict, rng: np.random.Generator) -> dict:
    """Uniform crossover between two configurations of the same family."""
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}
