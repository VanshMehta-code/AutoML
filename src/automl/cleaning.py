"""Metadata-driven cleaning and encoding in one fit/transform estimator.

Everything here is learned from the training fold only: winsorisation bounds,
one-hot levels, target-encoding statistics and imputation values. The fitted
object is serialised with the model, so inference applies exactly the same steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from .metadata import CATEGORICAL, CONSTANT, DATETIME, IDENTIFIER, NUMERIC

DT_PARTS = ("year", "month", "day", "dayofweek", "hour")
MISSING = "__missing__"


class Preprocessor(BaseEstimator, TransformerMixin):
    """Cleans and encodes a feature DataFrame using dataset metadata.

    Roles per column (decided from metadata):
      - constant / identifier  -> dropped
      - numeric                -> winsorised, median-imputed, missing-indicator added
      - datetime               -> expanded into year/month/day/dayofweek/hour
      - categorical (<= high_cardinality levels) -> one-hot (rare levels grouped)
      - categorical (>  high_cardinality levels) -> target encoded (cross-fitted)
    """

    def __init__(self, metadata, outlier_k=3.0, high_cardinality=20):
        self.metadata = metadata
        self.outlier_k = outlier_k
        self.high_cardinality = high_cardinality

    # ----- role assignment -------------------------------------------------
    def _assign_roles(self) -> None:
        cols = self.metadata["columns"]
        self.dropped_ = [c for c, i in cols.items() if i["kind"] in (CONSTANT, IDENTIFIER)]
        kept = {c: i for c, i in cols.items() if c not in self.dropped_}
        self.numeric_ = [c for c, i in kept.items() if i["kind"] == NUMERIC]
        self.datetime_ = [c for c, i in kept.items() if i["kind"] == DATETIME]
        cats = [c for c, i in kept.items() if i["kind"] == CATEGORICAL]
        self.low_cat_ = [c for c in cats if cols[c]["cardinality"] <= self.high_cardinality]
        self.high_cat_ = [c for c in cats if cols[c]["cardinality"] > self.high_cardinality]

    # ----- row-wise cleaning (uses fitted bounds) --------------------------
    def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for c in self.numeric_:
            lo, hi = self.bounds_[c]
            out[c] = pd.to_numeric(X[c], errors="coerce").astype(float).clip(lo, hi)
        for c in self.datetime_:
            d = pd.to_datetime(X[c], errors="coerce", format="mixed")
            for part in DT_PARTS:
                out[f"{c}__{part}"] = getattr(d.dt, part).astype(float)
        for c in self.low_cat_ + self.high_cat_:
            out[c] = X[c].astype("object").where(X[c].notna(), MISSING).astype(str)
        return pd.DataFrame(out, index=X.index)

    # ----- sklearn API -----------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        self._assign_roles()

        self.bounds_: dict[str, tuple[float, float]] = {}
        for c in self.numeric_:
            x = pd.to_numeric(X[c], errors="coerce").dropna()
            if len(x) and x.nunique() > 1:
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                self.bounds_[c] = (q1 - self.outlier_k * iqr, q3 + self.outlier_k * iqr)
            else:
                self.bounds_[c] = (-np.inf, np.inf)

        Xc = self._clean(X)
        self.numeric_cols_ = self.numeric_ + [f"{c}__{p}" for c in self.datetime_ for p in DT_PARTS]

        transformers = []
        if self.numeric_cols_:
            transformers.append((
                "num",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                self.numeric_cols_,
            ))
        if self.low_cat_:
            transformers.append((
                "low",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
                self.low_cat_,
            ))
        if self.high_cat_:
            transformers.append(("high", TargetEncoder(), self.high_cat_))
        if not transformers:
            raise ValueError("No usable features remain after cleaning; check the metadata.")

        self.ct_ = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
        self.ct_.fit(Xc, y)
        try:
            self.feature_names_ = [str(n) for n in self.ct_.get_feature_names_out()]
        except Exception:  # some transformer versions cannot name outputs; fall back to indices
            self.feature_names_ = [f"f{i}" for i in range(self.ct_.transform(Xc).shape[1])]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.ct_.transform(self._clean(X))
