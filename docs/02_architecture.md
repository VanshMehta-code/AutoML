# 02 Architecture

## The big picture

The system has two sides. **Training** is slow and searches many candidates. **Inference** is fast and runs one fixed path. The search only exists to produce a single artifact, `FittedModel`, which contains everything inference needs.

```
            TRAINING (slow, searches)                              INFERENCE (fast, fixed)
 ┌──────────────────────────────────────────────┐        ┌─────────────────────────────────────┐
 │ CSV ─► split ─► metadata ─► Preprocessor     │        │ raw DataFrame                        │
 │                              │               │        │   ─► schema check                    │
 │                  feature graph (MI, redund.) │        │   ─► Preprocessor.transform          │
 │                              │               │        │   ─► feature mask                    │
 │                  ACO feature subset          │ fit ─► │   ─► Σ weight × member.predict_proba │
 │                              │               │        │   ─► class probabilities             │
 │                  evolutionary search         │        └─────────────────────────────────────┘
 │                  (successive halving)        │                    ▲
 │                              │               │                    │ joblib
 │                  greedy ensemble (OOF)       │ ─── FittedModel ───┘
 │                              │               │
 │                  test-set score (once)       │
 └──────────────────────────────────────────────┘
```

Because `FittedModel` carries the fitted preprocessing, the mask, and the members together, a deployed model cannot disagree with training about how features were built.

## Module map

All code is in `src/automl/`.

| Module | Responsibility | Key public names |
|---|---|---|
| `config.py` | Load and validate YAML, reject unknown keys | `Config`, `SearchConfig`, `CleaningConfig` |
| `metadata.py` | Column kinds and statistics, on training rows only | `extract_metadata`, `column_kind` |
| `cleaning.py` | Winsorise, impute, expand dates, encode. One fit/transform estimator | `Preprocessor` |
| `feature_graph.py` | Relevance (mutual information) and redundancy (Spearman) between features | `build_feature_graph`, `FeatureGraph` |
| `models.py` | Model families, search spaces, sample / mutate / crossover | `REGISTRY`, `build`, `sample_params` |
| `evaluation.py` | Cross-validated AUC at a given row fraction and feature mask, with a cache | `Evaluator`, `Evaluation` |
| `evolution.py` | Generational search with successive halving and a time budget | `evolve` |
| `aco.py` | Ant colony search over feature subsets | `aco_select` |
| `ensemble.py` | Greedy ensemble selection on out-of-fold predictions | `greedy_ensemble` |
| `metrics.py` | ROC AUC for binary and multiclass | `auc_score` |
| `model.py` | The serialisable artifact and its inference path | `FittedModel` |
| `trainer.py` | Runs all stages in order | `AutoML` |
| `cli.py` | `automl train` and `automl predict` | `main` |
| `deploy/app.py` | HTTP service (FastAPI) around `FittedModel` | `app` |

## Dependency direction

Lower modules never import higher ones. `trainer.py` depends on everything. `model.py` depends on nothing except numpy, pandas and joblib. This keeps the artifact loadable in a serving container that does not need the search code. (The package is still installed as a whole; a slimmer serving package is on the [roadmap](09_roadmap.md).)

## Data flow and shapes

| Step | Input | Output |
|---|---|---|
| Split | `df` (n rows) | train (≈80%), test (≈20%), stratified |
| Preprocessor | train features (p columns, mixed types) | encoded matrix `X_enc` (n × q, all float) |
| Feature graph | `X_enc`, y | `relevance` (q), `redundancy` (q × q), `drop` (q) |
| Mask composition | graph mask (q), ACO mask (q′ ≤ q) | `full_mask` over the q encoded columns |
| Search | `X_sel` (n × q′) | list of full-fidelity evaluations with out-of-fold probabilities |
| Ensemble | OOF probabilities (n × classes) per member | weights summing to 1 |

Feature masks are always expressed over the encoded columns, so inference only needs `full_mask`.

## Where state lives

| State | Lives in | Persisted? |
|---|---|---|
| Metadata | `Preprocessor.metadata` | Yes, inside `model.joblib` |
| Winsorisation bounds, category levels, target-encoding stats | `Preprocessor` (fitted) | Yes |
| Feature mask | `FittedModel.mask` | Yes |
| Ensemble members and weights | `FittedModel` | Yes |
| Evaluation cache | `Evaluator._cache` | No. Lives only during one run |
| Search history | returned by `evolve` | Only the top members are kept |
| Run report | `FittedModel.report` and `report.json` | Yes |

See [03 Stages](03_stages.md) for what each step does and [08 Decisions](08_decisions.md) for why it is split this way.
