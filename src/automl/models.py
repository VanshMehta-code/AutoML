"""Model families, their search spaces, and the operators that evolve them.

Each family supports classification, regression, or both. families_for() keeps
only the configured families that can handle the current task.
Search-space entries are (kind, low, high) with kind in {"loguniform", "uniform", "int"}.
The "mlp" family is scikit-learn's multi-layer perceptron (a small neural network).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSIFICATION, REGRESSION = "classification", "regression"

REGISTRY: dict[str, dict] = {
    "hgb": {
        CLASSIFICATION: HistGradientBoostingClassifier,
        REGRESSION: HistGradientBoostingRegressor,
        "fixed": {"max_iter": 300, "early_stopping": True, "random_state": 0},
        "space": {
            "learning_rate": ("loguniform", 0.01, 0.3),
            "max_leaf_nodes": ("int", 8, 128),
            "min_samples_leaf": ("int", 5, 100),
            "l2_regularization": ("uniform", 0.0, 5.0),
        },
        "scale": False,
    },
    "rf": {
        CLASSIFICATION: RandomForestClassifier,
        REGRESSION: RandomForestRegressor,
        "fixed": {"n_estimators": 300, "n_jobs": -1, "random_state": 0},
        "space": {
            "max_depth": ("int", 3, 20),
            "min_samples_leaf": ("int", 1, 20),
            "max_features": ("uniform", 0.2, 1.0),
        },
        "scale": False,
    },
    "logreg": {
        CLASSIFICATION: LogisticRegression,
        REGRESSION: None,
        "fixed": {"max_iter": 2000},
        "space": {"C": ("loguniform", 1e-3, 1e2)},
        "scale": True,
    },
    "ridge": {
        CLASSIFICATION: None,
        REGRESSION: Ridge,
        "fixed": {},
        "space": {"alpha": ("loguniform", 1e-3, 1e3)},
        "scale": True,
    },
    "mlp": {
        CLASSIFICATION: MLPClassifier,
        REGRESSION: MLPRegressor,
        "fixed": {"max_iter": 300, "early_stopping": True, "random_state": 0},
        "space": {
            "hidden": ("int", 16, 128),  # one hidden layer of this width is searched; see build()
            "alpha": ("loguniform", 1e-5, 1e-1),
            "learning_rate_init": ("loguniform", 1e-4, 1e-2),
        },
        "scale": True,
    },
}


def _kind(task: str) -> str:
    return REGRESSION if task == "regression" else CLASSIFICATION


def families_for(task: str, requested: list[str]) -> list[str]:
    """Return the requested families that support the task. Raise if none do."""
    unknown = [f for f in requested if f not in REGISTRY]
    if unknown:
        raise KeyError(f"Unknown model families {unknown}. Known: {sorted(REGISTRY)}")
    usable = [f for f in requested if REGISTRY[f][_kind(task)] is not None]
    if not usable:
        raise ValueError(f"None of {requested} support task '{task}'.")
    return usable


def build(family: str, params: dict, task: str):
    """Instantiate a model for the task. Scale-sensitive families are wrapped with a scaler."""
    spec = REGISTRY[family]
    cls = spec[_kind(task)]
    if cls is None:
        raise ValueError(f"'{family}' does not support task '{task}'.")
    kwargs = {**spec["fixed"], **params}
    if family == "mlp":
        width = kwargs.pop("hidden")
        kwargs["hidden_layer_sizes"] = (width,)
    est = cls(**kwargs)
    return make_pipeline(StandardScaler(), est) if spec["scale"] else est


def default_params(family: str) -> dict:
    """Mid-range hyperparameters, used when a family must be run with fixed settings (e.g. ACO)."""
    out = {}
    for name, (kind, lo, hi) in REGISTRY[family]["space"].items():
        if kind == "loguniform":
            out[name] = float(np.sqrt(lo * hi))
        elif kind == "uniform":
            out[name] = float((lo + hi) / 2)
        else:
            out[name] = int((lo + hi) // 2)
    return out


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
