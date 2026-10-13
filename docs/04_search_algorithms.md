# 04 Search algorithms

Two searches run in sequence: ACO selects **which features** to use, then evolution selects **which model and hyperparameters** to use. Both use the same cached evaluator, so repeated candidates cost nothing.

## Shared machinery: the Evaluator

`evaluation.py` provides `Evaluator.evaluate(family, params, fraction, mask)`. It:

1. Applies the feature mask, if any
2. Subsamples rows to `fraction` of the training data (stratified)
3. Runs stratified k-fold cross-validation (`cv_folds`) with `predict_proba`
4. Returns the pooled out-of-fold AUC. At `fraction = 1.0` it also returns the out-of-fold probabilities, which the ensemble needs

Results are cached by a hash of (family, params, fraction, mask), so an identical candidate is never trained twice.

## Successive halving (the fidelity trick)

Most candidates are bad. Evaluating them on all rows is wasteful. So each generation:

1. **Screen:** every candidate is scored on `min_fraction` (default 20%) of the rows.
2. **Promote:** the top `max(elite, screened / eta)` candidates (default top third) are re-scored on 100% of the rows.

Only promoted candidates produce the full-data scores used for ranking and the ensemble. This is the main source of the speed gain described in [the evaluation notes](07_evaluation.md#speed).

## Evolutionary search

`evolution.py`, function `evolve`.

**Population.** `population` candidates per generation. Families are assigned round-robin, so every enabled family is always represented.

**Elite.** After each generation, the `elite` best full-data candidates across *all* generations are kept. They are the parents for the next generation.

**Breeding.** Each child is produced as follows:

- With 20% probability, a fresh random candidate from a random family. This keeps the search from converging too early.
- Otherwise, two elite parents are chosen at random.
  - If they share a family, a uniform crossover of their hyperparameters happens with 50% probability.
  - The result is mutated: each hyperparameter is resampled with probability 0.3.

**Stopping.** Stops after `generations`, or when `time_budget_seconds` is exceeded. The time check happens between evaluations, so the last evaluation can overrun slightly.

**Search spaces** are defined in `models.py` in the `REGISTRY`. Each entry is `(kind, low, high)`:

| Family | Hyperparameters searched |
|---|---|
| `hgb` (HistGradientBoostingClassifier) | `learning_rate` (log, 0.01–0.3), `max_leaf_nodes` (8–128), `min_samples_leaf` (5–100), `l2_regularization` (0–5) |
| `rf` (RandomForestClassifier) | `max_depth` (3–20), `min_samples_leaf` (1–20), `max_features` (0.2–1.0) |
| `logreg` (LogisticRegression, scaled) | `C` (log, 1e-3–1e2) |

| `ridge` (Ridge, scaled), regression only | `alpha` (log, 1e-3–1e3) |
| `mlp` (scikit-learn MLPClassifier / MLPRegressor, scaled), a one-hidden-layer neural network | `hidden` width (16–128), `alpha` (L2, log), `learning_rate_init` (log) |

`logreg` is classification-only and `ridge` is regression-only. `families_for()` filters the configured list down to the families that support the detected task, and fails if none do.

Fixed settings (not searched) are in the same registry entry, for example `max_iter` and `random_state`.

**Neural network note.** `mlp` is a small feed-forward network trained with scikit-learn. It is a real neural network, but not deep learning. PyTorch-based models are a possible later addition (see [09 Roadmap](09_roadmap.md)).

**Why evolution here:** the space mixes integers, continuous values, log scales and conditional families. Evolution handles that without gradients, and crossover tends to combine good settings found in different branches.

## Ant colony optimisation (ACO)

`aco.py`, function `aco_select`. Selects a **subset of features**, which is a combinatorial problem where ACO is a natural fit.

**Pheromone τ_i** per feature, initialised to 1. Accumulates on features that appear in good subsets.

**Heuristic η_i** per feature, from the feature graph: its mutual-information relevance, scaled to [0, 1], plus a tiny constant. Features that are individually informative are more likely to be tried.

**Each ant** includes feature i with probability

  p_i = w_i / (1 + w_i), where w_i = τ_i^α · η_i^β,

clipped to [0.05, 0.95]. An ant always includes at least one feature. Each ant's subset is scored with a fast model (logistic regression at `C = 1`) on `min_fraction` of the rows.

**Pheromone update per iteration:**

1. Evaporate: τ ← τ × (1 − ρ)
2. Deposit: each ant adds its normalised score to the features it used, where normalised means rescaled to [0, 1] across this iteration's ants
3. Clip τ to [0.01, 10] so no feature becomes permanently impossible or dominant

**Result.** The best subset seen across all ants and iterations. If that subset does not beat the score of using all graph-filtered features, ACO returns `None` and the full set is kept.

**Parameters:** `aco_ants`, `aco_iterations`, `aco_evaporation` (ρ), `aco_alpha` (α, pheromone weight), `aco_beta` (β, heuristic weight). Larger β favours the graph's relevance; larger α favours what has worked before.

## Ensemble selection

`ensemble.py`, function `greedy_ensemble`. This is Caruana-style greedy selection with replacement:

1. Start with an empty blend.
2. Add whichever top candidate most improves the blended out-of-fold AUC.
3. Repeat up to 25 times, stopping early when nothing improves.
4. Weights are the share of times each candidate was chosen.

Repeats are allowed, so a strong model can receive a larger weight. The result is usually better than picking the single best model, and it costs no extra training.

## Using the search results

| Question | Where to look |
|---|---|
| Which features survived? | `report.json` → `features_final`. The mask is stored in `FittedModel.mask` |
| Did ACO help? | Log line `[aco] ...`. `None` means no |
| Which models won? | `report.json` → `ensemble`, with each member's family, params, weight and CV AUC |
| Was the budget enough? | If `[evolve]` best scores still rise in the last generation, more generations or time may help. If they are flat, the search has converged |

## Tuning advice

- **Start with the defaults.** They are chosen to finish in a few minutes on typical tabular data.
- **More time before more generations.** A larger `time_budget_seconds` only helps if the search is still improving.
- **If ACO keeps returning `None`:** the features are either all useful or the graph has already removed the noise. Set `use_aco: false` to save time.
- **If `hgb` never wins but is slow:** remove it from `families` to save budget. A fast linear model may be enough.
