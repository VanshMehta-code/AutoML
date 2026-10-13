# 03 Stages

This page walks through the pipeline in the order it runs. For each stage: what it does, what it reads and writes, and the rule that prevents leakage.

**Golden rule:** anything learned from data (statistics, bounds, category levels, encodings, model weights) comes from the training split only. The test split is touched exactly once, at the end.

---

## Stage 1: Split

`trainer.py` splits the rows into train and test with `test_size` (default 0.2), stratified by target, seeded by `seed`. Rows with a missing target are dropped first, since they cannot be used for training or scoring.

Output: `train` and `test` DataFrames.

## Stage 2: Metadata

`metadata.py` describes the training data. Every later decision reads from this description rather than re-inspecting the data.

For each column it records:

| Field | Meaning |
|---|---|
| `kind` | `numeric`, `categorical`, `datetime`, `identifier`, or `constant` |
| `missing_frac` | Share of missing values |
| `cardinality` | Number of distinct non-missing values |
| `skew`, `kurtosis`, `outlier_frac` | Numeric columns only. Outliers are values outside 1.5 × IQR |

Kind assignment, in order:

1. One or fewer distinct values → `constant`
2. Boolean → `categorical`
3. Numeric dtype → `numeric`
4. Parses as dates (≥90% of sampled values) → `datetime`
5. More than 50 distinct values and at least 90% of rows → `identifier`
6. Everything else → `categorical`

Dataset-level: row and feature counts, task (`binary` or `multiclass`), class balance, and the smallest class share.

## Stage 3: Cleaning and encoding

`cleaning.py` provides `Preprocessor`, a single scikit-learn-style estimator that fits on training data and transforms any data the same way.

| Column kind | Treatment |
|---|---|
| `constant`, `identifier` | Dropped. Carries no signal, and identifiers would only memorise rows |
| `numeric` | Values outside `[Q1 − k·IQR, Q3 + k·IQR]` are clipped (`k = outlier_k`, default 3). Missing values are filled with the training median, plus a missing-indicator column if any were missing in training |
| `datetime` | Expanded into `year`, `month`, `day`, `dayofweek`, `hour` numeric columns |
| `categorical`, ≤ `high_cardinality` levels (default 20) | One-hot encoded. Levels rarer than 1% are grouped. Missing becomes its own level `__missing__`. Unseen levels at inference become all zeros |
| `categorical`, > `high_cardinality` levels | Target encoded (mean target per level, cross-fitted internally by scikit-learn to limit overfitting) |

Output: a float matrix with no missing values. Feature names are stored in `feature_names_`.

**Why no text handling yet:** free text longer than an identifier is currently treated as a high-cardinality categorical. That is a weak treatment. See [09 Roadmap](09_roadmap.md).

## Stage 4: Feature graph

`feature_graph.py` builds a small graph over the encoded features.

- **Relevance (node weight):** mutual information with the target, scaled to [0, 1]. Captures non-linear dependence, unlike correlation.
- **Redundancy (edge weight):** |Spearman rank correlation| between each pair. Rank-based, so it catches monotonic non-linear duplicates.

Redundant features are removed greedily. Features are visited most-relevant first. Any feature whose redundancy with a kept, more relevant feature is ≥ `redundancy_threshold` (default 0.95) is dropped.

Relevance also feeds ACO as its heuristic desirability.

**Cost:** redundancy is computed for all pairs, so it grows with the square of the feature count. Fine up to a few thousand encoded columns. Beyond that, see the roadmap.

## Stage 5: ACO feature selection

Optional (`search.use_aco`). Runs on the features kept by Stage 4. Details in [04 Search algorithms](04_search_algorithms.md#ant-colony-optimisation-aco).

It returns a subset mask, or `None` if no subset beats using all features. In that case the pipeline keeps everything. ACO can therefore never make the final model worse than the graph-filtered baseline on its own scoring.

## Stage 6: Evolutionary hyperparameter search

Searches over model families (`hgb`, `rf`, `logreg` by default) and their hyperparameters. Uses successive halving so most candidates are only evaluated on a small share of rows. Details in [04 Search algorithms](04_search_algorithms.md#evolutionary-search).

Output: every candidate that reached full data, with its cross-validated score and out-of-fold probabilities, best first.

## Stage 7: Ensemble and refit

- Take the top `ensemble_size` candidates by CV score.
- Choose blend weights with greedy ensemble selection on their out-of-fold probabilities. Members that get weight zero are dropped.
- Refit each kept member on **all** training rows, with its hyperparameters fixed.

The refit is what makes the final model use 100% of the training data rather than 80% of it per fold.

## Stage 8: Test score and packaging

The blended members predict the held-out test set once. The test AUC and accuracy are written to the report. The test set does not influence any earlier decision.

Then `FittedModel` is assembled (preprocessor, mask, members, weights, input columns, report) and returned for saving.

---

## Leakage: what is protected and what is not

| Item | Status |
|---|---|
| Test set influences search, cleaning or ensemble | **Protected.** Test is used only in Stage 8 |
| Metadata and bounds come from training rows | **Protected** |
| Cross-validated scores are fully independent of each fold's preprocessing | **Not fully.** The Preprocessor is fitted once on all training rows, and the CV folds reuse that encoding. The effect is small for one-hot and median imputation, and target encoding is cross-fitted internally, but CV scores can be slightly optimistic. **Use the test AUC as the honest number.** Fitting preprocessing inside each fold is on the [roadmap](09_roadmap.md) |
| Ensemble weights come from out-of-fold predictions | **Protected.** OOF predictions never see their own row's label |
