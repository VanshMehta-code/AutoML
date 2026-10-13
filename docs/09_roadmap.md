# 09 Roadmap

What is built, what is partial, and what is planned, in priority order. Status is checked against the code, not intent.

## Status today

| Area | Status | Notes |
|---|---|---|
| Config loading with strict keys | Implemented | Ranges are not validated yet |
| Metadata extraction | Implemented | Column kinds and statistics, training rows only |
| Cleaning and encoding (`Preprocessor`) | Implemented | Numeric, categorical, datetime, identifiers, constants |
| Feature graph (MI + Spearman redundancy) | Implemented | Quadratic in feature count |
| ACO feature selection | Implemented | Simplified ACO; uses logistic regression as the scorer |
| Evolutionary hyperparameter search | Implemented | Successive halving, time budget, caching |
| Greedy ensemble | Implemented | On out-of-fold predictions |
| Test evaluation and report | Implemented | `report.json` |
| CLI (`train`, `predict`) | Implemented | |
| HTTP service | Implemented | No authentication yet |
| Binary classification | Tested | Covered by the test suite |
| Multiclass classification | Tested | One end-to-end test; broader coverage is step 1 below |
| Regression | Implemented | Ridge, gradient boosting, random forest and MLP regressors; scored by R² |
| Automatic leakage screening | Implemented | Single-feature screen, drops at 0.95 by default. Moderate and combination leaks still need review |
| Neural network family | Implemented | scikit-learn MLP (one hidden layer); not deep learning |
| Train/test gap reporting | Implemented | Reported with saved splits |
| Preprocessing fitted inside each CV fold | Planned | Removes the small CV optimism |
| Gradient boosting families (LightGBM, XGBoost) | Planned | |
| Genetic programming for feature generation | Planned | Evolve features such as ratios and logs |
| Latency in the search objective | Planned | Multi-objective (NSGA-II) on AUC and latency |
| Distilled single model | Planned | Mimic the ensemble with one fast model |
| Probability calibration | Planned | |
| Drift monitoring | Planned | Built on stored training metadata |
| Benchmark scripts (OpenML) | Planned | |
| Experiment tracking (MLflow) | Planned | |
| Full-data refit for production | Planned | Train on train + test after reporting |
| Text feature handling | Planned | Currently treated as a categorical or dropped |

## Next steps, in priority order

### 1. Verify what exists (now)

- Extend the multiclass tests: check every family trains and scores correctly on 3+ classes, and that ACO works on multiclass data.
- Run the ablation table from [07 Evaluation](07_evaluation.md#ablation-does-each-component-earn-its-place) on two or three real datasets. This tells us whether ACO and the graph earn their cost. Nothing else should be built on them until this is known.
- Add config range validation (`eta ≥ 2`, `elite ≤ population`, `0 < test_size < 1`).
- Test the leakage screen on a real dataset with a known leak, and measure how often it misses one.
- Add a leakage check against the *target itself*: flag any feature that is a deterministic function of the target (for example a category derived from a regression target).

### 2. Make CV scores honest

- Move preprocessing inside each CV fold, so the search's scores carry no preprocessing optimism. This costs more compute per fold, so measure the trade-off with the successive-halving budget.

### 3. Benchmark against baselines

- Script that runs AutoML and baselines (LightGBM defaults, Optuna TPE, FLAML if installed) on OpenML-CC18 datasets under one time budget, with repeated seeds.
- Output: mean rank, mean test AUC, spread, and seconds per dataset.

### 4. Add stronger model families

- LightGBM (and XGBoost) as optional families. Each is one `REGISTRY` entry plus an optional dependency. Expect this to be the single largest accuracy and speed gain.

### 5. Deployment hardening

- Multi-objective search: NSGA-II over AUC and prediction latency (measured on the target machine), plus model size.
- Distillation: train one small model to mimic the ensemble's probabilities, and compare its AUC against the ensemble.
- Probability calibration on a held-out split, with reliability plots in the report.
- Slim serving package containing only inference code, so the container does not need the search code.
- Drift monitor: compare live input statistics with stored training metadata, and alert on the thresholds in [06 Deployment](06_deployment.md#monitoring-in-production).
- Service authentication and request-size limits.

### 6. Feature generation

- Genetic programming over feature expressions (ratios, differences, logs, group aggregations), with CV improvement as fitness and the feature graph guiding which pairs to combine.
- Only after step 1 shows that feature work pays off on the target data.

### 7. Scale and data types

- Redundancy computation in blocks or on a sample, for wide data.
- Text features via TF-IDF or sentence embeddings, inside the Preprocessor.
- Surrogate-assisted search (a cheap model that predicts which candidates are worth evaluating), for large budgets.
- Asynchronous, steady-state evolution across workers.

### 8. Refit and release

- Option to refit the final model on train + test after reporting, with a clear note that the reported score then refers to the earlier split.
- Versioned artifact directories and a `CHANGELOG.md` generated per release.

## Rules for working on this roadmap

- Do not start a later step until the earlier one has a measured result.
- Every new component needs an ablation showing it helps. If it does not, it is removed or left out.
- New decisions go in [08 Decisions](08_decisions.md). Update the status table here when work lands.
