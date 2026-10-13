"""The fitted model artifact: everything inference needs, in one serialisable object.

Inference path: raw DataFrame -> schema check -> Preprocessor -> feature mask
-> weighted sum of ensemble members' outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass
class FittedModel:
    preprocessor: object
    task: str
    classes: np.ndarray | None  # None for regression
    mask: np.ndarray
    members: list
    weights: np.ndarray
    input_columns: list[str]
    target: str
    report: dict = field(default_factory=dict)

    def check_input(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.input_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        self.check_input(df)
        return self.preprocessor.transform(df[self.input_columns])[:, self.mask]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            raise ValueError("predict_proba is only for classification; use predict() for regression.")
        X = self._transform(df)
        return sum(w * m.predict_proba(X) for w, m in zip(self.weights, self.members))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._transform(df)
        if self.task == "regression":
            return sum(w * m.predict(X) for w, m in zip(self.weights, self.members))
        proba = sum(w * m.predict_proba(X) for w, m in zip(self.weights, self.members))
        return self.classes[proba.argmax(axis=1)]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "FittedModel":
        return joblib.load(path)
