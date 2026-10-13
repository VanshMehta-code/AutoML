# 08 Decisions

A log of the significant design choices, with the reasoning and the trade-off accepted. New decisions go at the bottom. Never edit an old decision in place. Add a new one that supersedes it.

Format: **Context** (why the question came up), **Decision**, **Consequences** (what we gain and what we give up).

---

### D-001: Metadata computed on the training split only

**Context.** Metadata includes statistics (medians, quartiles, class balance) that would leak test information if computed on all rows.

**Decision.** `extract_metadata` runs on the training split. The Preprocessor fits on training rows too.

**Consequences.** Clean test evaluation. Slightly less information for rare categories, which only appear in test. Those are handled as unseen levels at inference, which is the same situation as production.

---

### D-002: One fit/transform estimator for all cleaning and encoding

**Context.** Cleaning rules must be identical in training and serving. Separate scripts for each drift apart.

**Decision.** `Preprocessor` is a single estimator. It learns bounds, level sets, and encodings in `fit`, and applies them in `transform`. It is serialised with the model.

**Consequences.** Training and serving cannot disagree. Inference needs the same package version (see [06 Deployment](06_deployment.md#what-you-deploy)). Changing a cleaning rule means retraining.

---

### D-003: Identifiers and constant columns are dropped, not encoded

**Context.** An ID column has one distinct value per row. One-hot or target encoding on it memorises the training rows and generalises to nothing.

**Decision.** A column is dropped when it is constant, or when it is non-numeric with more than 50 distinct values covering at least 90% of rows.

**Consequences.** Simple and safe. A genuinely informative high-cardinality column that happens to be unique per row is also dropped, which is rare in practice. Free text is treated the same way when it is unique, so text signal is currently lost (see roadmap).

---

### D-004: Target encoding for high-cardinality categoricals

**Context.** One-hot on a column with 500 levels creates 500 mostly-empty columns, which is slow and overfits. Dropping it loses real signal.

**Decision.** Above `high_cardinality` levels (default 20), use scikit-learn's `TargetEncoder`, which cross-fits internally.

**Consequences.** Keeps the signal in one column per class. Requires the target at fit time, which is always available in training. Works for binary and multiclass.

---

### D-005: The feature graph uses mutual information and rank correlation

**Context.** We want to remove duplicate features and rank features by relevance, cheaply, and without assuming linearity.

**Decision.** Relevance is mutual information with the target. Redundancy is |Spearman| between features, which is rank-based and so catches monotonic non-linear duplicates.

**Consequences.** Robust to non-linear relationships. Redundancy is quadratic in feature count, so very wide data (thousands of columns) will need a different approach (see roadmap).

---

### D-006: Successive halving before evolution, not full evaluation for every candidate

**Context.** The main compute cost is cross-validated training. Most random candidates are poor, so scoring them on all rows wastes most of the budget.

**Decision.** Screen every candidate at 20% of rows. Promote the top third to full data. Only full-data results enter the ranking and the ensemble.

**Consequences.** Far fewer full-data trainings per generation, so more candidates fit in the same budget. The speedup has not been measured yet. The screening score is a noisier proxy, so a strong late bloomer can be missed. The next generation partly compensates, since it re-evaluates the elite. Measure the gain on real data before relying on it (see [07 Evaluation](07_evaluation.md#speed)).

---

### D-007: Evolution for hyperparameters, ACO for feature subsets

**Context.** The two problems have different shapes. Hyperparameters are continuous, integer and conditional on family. Feature selection is a binary subset choice over a graph-structured heuristic.

**Decision.** Evolutionary search (generational, with crossover and mutation) for hyperparameters and family. ACO with graph relevance as heuristic for the feature subset.

**Consequences.** Each method fits its problem. Both are more code than a single optimiser. ACO can return no improvement, and the pipeline handles that by keeping all features. Ablation is required to confirm each one helps (see [07 Evaluation](07_evaluation.md#ablation-does-each-component-earn-its-place)).

---

### D-008: Ensemble by greedy selection on out-of-fold predictions

**Context.** Averaging the top models usually helps. Choosing weights by optimising on the test set would be cheating.

**Decision.** Caruana-style greedy selection with replacement, using out-of-fold predictions on training data only. Members are refit on all training rows.

**Consequences.** The ensemble gets full training data and an honest weight choice, at no extra search cost. The final artifact is larger and slower than a single model. Latency-sensitive use should cap `ensemble_size`.

---

### D-009: Refit on all training rows after selection

**Context.** Cross-validation trains each model on 80% of training rows. Refitting on 100% usually gives a small, real gain.

**Decision.** Each kept member is refit on all training rows with its hyperparameters fixed.

**Consequences.** The shipped model uses all training data. The test set is still held out and is not used. Training on train + test for the final production model is a further option, noted in the roadmap.

---

### D-010: Test set held out once, reported once

**Context.** Any repeated use of the test set for choices turns it into a validation set and inflates the score.

**Decision.** The test split is created at the start, used only to compute the final AUC and accuracy, and never used to choose anything.

**Consequences.** The reported test AUC is honest. Users must still avoid retuning many configs against it. Use repeated seeds for decisions (see [07 Evaluation](07_evaluation.md#comparing-two-runs-fairly)).

---

### D-011: Feature masks are expressed over encoded columns

**Context.** Feature selection happens after encoding, so one-hot and target-encoded columns are separate features.

**Decision.** Masks index the encoded matrix. `FittedModel.mask` is over encoded columns, and inference applies it after the Preprocessor.

**Consequences.** Simple and exact. Feature reports refer to encoded names, which can be less readable than the original column names. `feature_names_` gives the mapping.

---

### D-012: Plain scikit-learn models for the first version

**Context.** Gradient boosting libraries (LightGBM, XGBoost, CatBoost) are usually faster and stronger than scikit-learn's histogram boosting on large data.

**Decision.** Start with scikit-learn only (`HistGradientBoostingClassifier`, `RandomForestClassifier`, `LogisticRegression`). This keeps installation simple and dependencies few.

**Consequences.** Fewer dependencies and easier deployment. Lower ceiling on speed and accuracy on large datasets. Adding LightGBM as an optional family is a planned change (see roadmap). It is one entry in `REGISTRY`, plus its dependency.

---

### D-013: Config rejects unknown keys

**Context.** A misspelled key that is silently ignored leads to runs that look configured but are not.

**Decision.** Unknown keys at the top level, and inside `search` and `cleaning`, raise an error.

**Consequences.** Typos fail early. Config files cannot carry notes as extra keys. Use YAML comments for notes.

---

### D-014: Documentation lives in `docs/`, with code comments kept short

**Context.** Code should explain itself through names and structure. Long comments go stale.

**Decision.** Each module has a short docstring stating its purpose and the key rule it enforces. Explanations live in `docs/`, each file answering one kind of question.

**Consequences.** Docs can be read without opening code. When behaviour changes, the docs must change too. Treat a doc that disagrees with the code as a bug.
