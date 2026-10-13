"""Typed configuration loaded from YAML. Unknown keys raise, so typos fail loudly."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

TASKS = ("auto", "binary", "multiclass", "regression")
LEAK_ACTIONS = ("drop", "warn")


@dataclass
class LeakageConfig:
    action: str = "drop"          # drop: remove flagged columns; warn: report only
    threshold: float = 0.95       # single-column score at or above this is a leak
    warn_threshold: float = 0.85  # at or above this (but below threshold): review manually


@dataclass
class CleaningConfig:
    outlier_k: float = 3.0
    high_cardinality: int = 20
    redundancy_threshold: float = 0.95


@dataclass
class SearchConfig:
    families: list[str] = field(default_factory=lambda: ["hgb", "rf", "logreg", "ridge", "mlp"])
    population: int = 16
    generations: int = 4
    elite: int = 4
    min_fraction: float = 0.2
    eta: int = 3
    time_budget_seconds: int = 900
    use_aco: bool = True
    aco_ants: int = 8
    aco_iterations: int = 6
    aco_evaporation: float = 0.2
    aco_alpha: float = 1.0
    aco_beta: float = 2.0
    ensemble_size: int = 5


@dataclass
class Config:
    target: str | None = None
    task: str = "auto"
    drop_columns: list[str] = field(default_factory=list)
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    leakage: LeakageConfig = field(default_factory=LeakageConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got '{self.task}'")
        if self.leakage.action not in LEAK_ACTIONS:
            raise ValueError(f"leakage.action must be one of {LEAK_ACTIONS}, got '{self.leakage.action}'")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        allowed = {"target", "task", "drop_columns", "seed", "test_size", "cv_folds",
                   "leakage", "cleaning", "search"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown config key(s): {sorted(unknown)}. Allowed: {sorted(allowed)}")
        return cls(
            target=raw.get("target"),
            task=raw.get("task", "auto"),
            drop_columns=raw.get("drop_columns", []) or [],
            seed=raw.get("seed", 42),
            test_size=raw.get("test_size", 0.2),
            cv_folds=raw.get("cv_folds", 5),
            leakage=LeakageConfig(**raw.get("leakage", {})),
            cleaning=CleaningConfig(**raw.get("cleaning", {})),
            search=SearchConfig(**raw.get("search", {})),
        )
