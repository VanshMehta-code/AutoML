# 07 Evaluation

How results are measured, what each number means, and how to tell a real improvement from noise. Read this before you trust a score or compare two runs.

## Metrics

| Metric | Where it is used | Meaning |
|---|---|---|
| **ROC AUC** | Everything: search, ensemble, test | Probability that a random positive is ranked above a random negative. 0.5 is chance, 1.0 is perfect. For multiclass, one-vs-rest averaged |
| **Accuracy** | Reported on test only | Share of correct top-class predictions. Depends on the decision threshold and class balance, so compare it only against the majority-class baseline |
| **CV AUC** (`cv_auc`, `best_cv_auc`) | Search ranking | Out-of-fold AUC from `cv_folds` folds on the training split |
| **Test AUC** (`test_auc`) | The final honest score | AUC on the held-out split. Never used for any decision |

AUC is the optimisation target because it does not depend on a threshold. Accuracy is reported for readability only.

## Reading the run report

```json
{
  "task": "binary",
  "n_train": 960,  "n_test": 240,
  "dropped_columns": ["const", "uid"],
  "features_encoded": 13,   "features_after_graph": 12,   "features_final": 2,
  "candidates_evaluated": 23,
  "best_cv_auc": 0.9442,
  "test_auc": 0.9227,  "test_accuracy": 0.8333,
  "ensemble": [{"family": "logreg", "params": {"C": 0.37}, "weight": 1.0, "cv_auc": 0.9442}],
  "seconds": 24.0
}
```

| Field | How to read it |
|---|---|
| `dropped_columns` | Constant or identifier columns removed. If a column you expected is here, check the data |
| `features_encoded` → `features_after_graph` → `features_final` | How many features existed, survived redundancy removal, and survived ACO. A large drop at ACO with no accuracy loss is a good sign |
| `candidates_evaluated` | Total candidates that reached the search. Compare with the budget |
| `best_cv_auc` vs `test_auc` | The gap estimates optimism from the search. See the next section |
| `ensemble` | Members, weights and their CV AUCs. Weights sum to 1 |
| `seconds` | Wall-clock training time |

## Overfitting to the search

Searching many candidates and picking the best one by CV score makes that score optimistic. This is the main honest-evaluation risk in any AutoML system.

- **Small gap** (`best_cv_auc − test_auc` under about 0.02): the search generalises.
- **Large gap** (above about 0.05): the search has overfitted the CV folds. Remedies: more data, fewer generations, a larger `cv_folds`, or a smaller search space. Trust `test_auc`, not `best_cv_auc`.

The test AUC is a single number from one split, so it also has noise. With a test set of a few hundred rows, differences below about 0.02 AUC are not meaningful.

## Comparing two runs fairly

A new run beats an old one only if all of these hold:

1. **Same data and same test split.** Keep `seed` and `test_size` fixed.
2. **Same time budget**, or report both the score and the seconds.
3. **Compare on test AUC**, not CV AUC.
4. **Repeat with several seeds.** Run at least 3–5 seeds and report the mean and spread. One lucky seed can look like a real gain.

A quick loop for repeated runs:

```bash
for s in 1 2 3 4 5; do
  sed "s/seed: 42/seed: $s/" configs/default.yaml > /tmp/seed_$s.yaml
  automl train --data data/train.csv --target y --config /tmp/seed_$s.yaml --out runs/seed_$s
done
python -c "import json,glob,statistics as st; a=[json.load(open(f))['test_auc'] for f in glob.glob('runs/seed_*/report.json')]; print(f'test AUC mean={st.mean(a):.4f} sd={st.stdev(a):.4f}')"
```

## Benchmarking against baselines

Accuracy is only meaningful relative to something. Compare the AutoML result against:

| Baseline | Why |
|---|---|
| Majority class | Floor for accuracy. Always beat it |
| Logistic regression, default settings | Floor for AUC |
| LightGBM / XGBoost / CatBoost, default settings | Strong untuned reference |
| Optuna (TPE) with the same budget | Strong tuned reference |
| FLAML or AutoGluon with the same budget | Full AutoML reference |

Compare under the **same wall-clock time budget**. A method that wins with 10× the compute has not won in any practical sense.

Recommended public benchmark: OpenML-CC18 (a suite of classification datasets). Use at least 10 datasets, report mean rank and mean AUC per dataset with spread, and include the time per dataset. Benchmark scripts are on the [roadmap](09_roadmap.md). Until then, do this by hand with the CLI.

## Ablation: does each component earn its place?

For a convincing result, switch each stage off and measure the drop:

| Run | Change from default |
|---|---|
| Full | none |
| No ACO | `use_aco: false` |
| No graph | `redundancy_threshold: 1.0` (removes nothing) |
| No ensemble | `ensemble_size: 1` |
| Single family | `families: [hgb]` |
| No search (defaults only) | `population: 1, generations: 1` |

If a component does not improve test AUC beyond noise, remove it. A simpler system is faster, cheaper, and easier to deploy.

## Speed

Speed in this project comes from **not wasting evaluations**, which matters more than a faster algorithm:

- **Successive halving:** most candidates are scored on 20% of rows. Only the best reach full data.
- **Shared preprocessing:** cleaning is fitted once per run, not once per candidate.
- **Caching:** identical (model, params, rows, features) combinations are never retrained.
- **Early stopping** inside `hgb` stops boosting when validation scores plateau.

Where time goes, with default settings: each generation screens 16 candidates at 20% of rows (about 3 full-data equivalents), then promotes about 5 candidates to full data (about 5 full-data equivalents). The promoted full-data passes therefore cost more than screening, even though screening touches more candidates. The ACO pass (with its own cheap logistic-regression scoring) and the final refit are smaller. Check the real split by timing the stages in your own runs.

To measure speed: compare `seconds` in `report.json` at a fixed `time_budget_seconds`, and read `candidates_evaluated`. More candidates per second at equal AUC is a real gain.

## Known evaluation limits

- CV scores are slightly optimistic because preprocessing is fitted once on all training rows (see [03 Stages, leakage](03_stages.md#leakage-what-is-protected-and-what-is-not)). Test AUC is the trustworthy number.
- The test set is one split. Use repeated seeds for decisions (see above).
- Regression is not supported, so there are no RMSE or MAE numbers.
- Calibration (whether 0.7 means 70%) is not checked. Do this before using probabilities as costs or risk scores.
