"""Dataset metadata: column roles and statistics that every later stage reads.

Metadata is always computed on the training split only, so nothing from the
test set can influence cleaning decisions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

# Column kinds produced by column_kind()
NUMERIC, CATEGORICAL, DATETIME, IDENTIFIER, CONSTANT = (
    "numeric",
    "categorical",
    "datetime",
    "identifier",
    "constant",
)


def _looks_like_datetime(s: pd.Series) -> bool:
    if ptypes.is_datetime64_any_dtype(s):
        return True
    if not (ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def column_kind(s: pd.Series, n_rows: int) -> str:
    """Classify one column. Order matters: constant first, identifiers before categoricals."""
    n_unique = s.nunique(dropna=True)
    if n_unique <= 1:
        return CONSTANT
    if ptypes.is_bool_dtype(s):
        return CATEGORICAL
    if ptypes.is_numeric_dtype(s):
        return NUMERIC
    if _looks_like_datetime(s):
        return DATETIME
    if n_unique > 50 and n_unique >= 0.9 * n_rows:
        return IDENTIFIER
    return CATEGORICAL


def extract_metadata(df: pd.DataFrame, target: str) -> dict:
    """Return a JSON-serialisable description of the dataset and each feature."""
    X = df.drop(columns=[target])
    y = df[target]
    n_rows = len(df)

    class_frac = y.value_counts(normalize=True)
    task = "binary" if y.nunique() == 2 else "multiclass"

    columns: dict[str, dict] = {}
    for name in X.columns:
        s = X[name]
        kind = column_kind(s, n_rows)
        info: dict = {
            "kind": kind,
            "missing_frac": float(s.isna().mean()),
            "cardinality": int(s.nunique(dropna=True)),
        }
        if kind == NUMERIC:
            x = s.dropna().astype(float)
            if len(x) > 2:
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() if iqr > 0 else 0.0
                info.update(
                    skew=float(x.skew()),
                    kurtosis=float(x.kurt()),
                    outlier_frac=float(outliers),
                )
        columns[name] = info

    return {
        "n_rows": int(n_rows),
        "n_features": int(X.shape[1]),
        "task": task,
        "n_classes": int(y.nunique()),
        "class_balance": {str(k): float(v) for k, v in class_frac.items()},
        "min_class_frac": float(class_frac.min()),
        "missing_frac_total": float(np.mean([c["missing_frac"] for c in columns.values()])) if columns else 0.0,
        "columns": columns,
    }
