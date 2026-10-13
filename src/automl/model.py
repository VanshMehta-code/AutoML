"""The fitted model artifact: everything inference needs, in one serialisable object.

Inference path: raw DataFrame -> schema check -> Preprocessor -> feature mask
-> weighted sum of ensemble members' probabilities.
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
    classes: np.ndarray
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

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self.check_input(df)
        X = self.preprocessor.transform(df[self.input_columns])[:, self.mask]
        return sum(w * m.predict_proba(X) for w, m in zip(self.weights, self.members))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.classes[self.predict_proba(df).argmax(axis=1)]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "FittedModel":
        return joblib.load(path)
