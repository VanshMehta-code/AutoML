# 11 Troubleshooting

Each entry gives the error or symptom, the usual cause, and the fix.

## Setup

**`automl: command not found`**
The package is not installed in the active environment. Activate your venv and run `pip install -e ".[dev,serve]"` from the project root.

**`ModuleNotFoundError: No module named 'automl'`** when running tests or `deploy/app.py`
Run from the project root after `pip install -e .`. The `src/` layout needs the editable install.

**`ModuleNotFoundError: No module named 'fastapi'`**
Install the serving extra: `pip install -e ".[serve]"`.

## Configuration

**`ValueError: Unknown config key(s): [...]`**
A top-level key is misspelled or not supported. The message lists the allowed keys.

**`TypeError: SearchConfig.__init__() got an unexpected keyword argument ...`**
A key under `search:` or `cleaning:` is misspelled. Check [05 Configuration](05_configuration.md).

**`ValueError: config.target must be set`**
Set `target:` in the YAML or pass `--target column_name`.

**`ValueError: Target column 'x' not found`**
The name does not match the CSV header exactly. Names are case-sensitive, and stray spaces count.

**`Unknown model family 'xgb'`**
Only `hgb`, `rf` and `logreg` exist today. See [09 Roadmap](09_roadmap.md) for planned families.

## Data

**`ValueError: stratify ...` / `The least populated class in y has only 1 member`**
A class has too few rows to split. Check `class_balance` in the metadata. Merge or remove the very rare class, or collect more rows.

**`ValueError: No usable features remain after cleaning`**
Every column was constant or an identifier. Check that the target is correct and the columns have variation.

**A column you expected is missing from `dropped_columns`... or present when it should not be**
Column kinds are inferred. A numeric code stored as text may parse as dates, or a unique text field may be dropped as an identifier. Check `metadata.columns[name]["kind"]` in the training log or report.

**Dates are not recognised**
Detection requires at least 90% of sampled values to parse as dates. Mixed or unusual formats fall below that. Convert to ISO format (`2024-03-15`) before training.

## Training

**`RuntimeError: Search produced no results; increase time_budget_seconds.`**
The time budget ran out before any candidate reached full data. Raise `time_budget_seconds`, or reduce `population`.

**Search finishes instantly with very few evaluations**
The budget is small, or `generations` is 1. Check `candidates_evaluated` in the report.

**`[aco] no subset beat all features; keeping them`**
Expected behaviour. Not an error. ACO found no improvement over using every feature.

**Training is slow**
Lower `cv_folds`, drop `rf` from `families` (it is often the slowest), reduce `population`, or set `use_aco: false`. Then measure which stage dominates.

**Test AUC is much lower than `best_cv_auc`**
Probably overfitting to the search. See [07 Evaluation](07_evaluation.md#overfitting-to-the-search). Use more data, fewer generations, or a smaller search space.

**Results change between runs**
Keep `seed` fixed. Time-budget cutoffs also make results depend on machine speed, so a slower machine can evaluate fewer candidates. Compare runs on the same hardware.

## Prediction

**`ValueError: Input is missing required columns: [...]`** (HTTP 422 from the service)
The input lacks columns the model was trained on. The list names them. Send every column in `input_columns`, even if some are unusual.

**`ValueError: could not convert string to float`** during prediction
A value in a numeric column is text that cannot be parsed. Check the input for stray text or units such as `"12 kg"`.

**Probabilities do not sum to 1**
They should always sum to 1 (up to rounding). If not, check that `weights` in the loaded model sum to 1 and that you are not mixing artifacts.

**Predictions differ between training and serving**
Check that the serving environment has the same scikit-learn and numpy versions as training. Compare `requirements.txt` with the live environment.

**`InconsistentVersionWarning` when loading `model.joblib`**
The scikit-learn version differs from the one used for training. Install the recorded version, or retrain in the serving environment.

## Service

**Service starts but `/predict` returns 500**
Look at the server log. Usually a malformed JSON body. The body must be a JSON list of objects, for example `[{"x1": 1.0, ...}]`.

**`KeyError: 'MODEL_PATH'`** or `FileNotFoundError` at startup
Set `MODEL_PATH` to the full path of an existing `model.joblib`.

## Still stuck?

1. Re-run with the quick-experiment profile from [05 Configuration](05_configuration.md#example-profiles) to see whether the problem is data or budget.
2. Run `pytest -q`. If the tests fail, the install or the environment is the problem, not your data.
3. Check `report.json` to see which stage the run reached.
