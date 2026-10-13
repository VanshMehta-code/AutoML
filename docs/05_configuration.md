# 05 Configuration

Configuration lives in a YAML file. The default is `configs/default.yaml`. Pass another file with `--config path.yaml`. Any key you omit takes the default from `src/automl/config.py`.

Unknown keys are rejected with an error, so a typo such as `generatons` fails immediately rather than being ignored.

## Top-level keys

| Key | Default | Range | What it does |
|---|---|---|---|
| `target` | `null` | a column name | The column to predict. Required. Can be overridden with `--target` |
| `seed` | `42` | any integer | Controls the split, CV folds, subsampling, feature-graph randomness, and search. Same seed plus same data gives the same result |
| `test_size` | `0.2` | 0.05–0.4 | Share of rows held out for the final honest score. Smaller gives more training data but a noisier test score |
| `cv_folds` | `5` | 3–10 | Folds for cross-validated scoring during search. More folds means less noisy scores and slower search |

## `cleaning`

| Key | Default | Range | What it does |
|---|---|---|---|
| `outlier_k` | `3.0` | 1.5–5 | Numeric values beyond `k × IQR` from the quartiles are clipped. Lower clips more aggressively |
| `high_cardinality` | `20` | 5–200 | Categoricals with more distinct levels than this use target encoding instead of one-hot. Raise it if one-hot is affordable and you want exact level effects |
| `redundancy_threshold` | `0.95` | 0.8–1.0 | Two features with |Spearman| at or above this are treated as duplicates, and the less relevant one is dropped. Lower removes more |

## `search`

| Key | Default | Range | What it does |
|---|---|---|---|
| `families` | `[hgb, rf, logreg]` | any of `hgb`, `rf`, `logreg` | Model families the search may use. Removing one saves budget |
| `population` | `16` | 4–100 | Candidates per generation |
| `generations` | `4` | 1–50 | Generations to run before stopping (unless time runs out first) |
| `elite` | `4` | 1–population | Best candidates kept as parents each generation. Also the minimum number promoted to full data each generation |
| `min_fraction` | `0.2` | 0.05–1.0 | Row share used for cheap screening. Lower is faster but noisier for ranking |
| `eta` | `3` | 2–5 | Successive-halving factor. Top `1/eta` of screened candidates are promoted to full data |
| `time_budget_seconds` | `900` | 30 and up | Hard cap on search time. Checked between evaluations |
| `use_aco` | `true` | true/false | Run ACO feature selection |
| `aco_ants` | `8` | 2–50 | Ants per ACO iteration |
| `aco_iterations` | `6` | 1–50 | ACO iterations |
| `aco_evaporation` | `0.2` | 0.05–0.5 | Pheromone evaporation rate ρ per iteration. Higher forgets faster |
| `aco_alpha` | `1.0` | 0–3 | Weight of pheromone (past success) in feature choice |
| `aco_beta` | `2.0` | 0–5 | Weight of the graph's relevance heuristic. Higher trusts mutual information more |
| `ensemble_size` | `5` | 1–20 | Top candidates considered for the blend. Only members with non-zero weight are kept |

## Example profiles

**Quick experiment** (a few minutes, for checking that data loads and the target is right):

```yaml
search:
  families: [logreg, hgb]
  population: 8
  generations: 2
  time_budget_seconds: 120
  use_aco: false
```

**Balanced default** (the shipped `configs/default.yaml`, about 15 minutes):

as shipped.

**Thorough overnight run** (more search, same code path):

```yaml
search:
  population: 40
  generations: 12
  time_budget_seconds: 14400
  ensemble_size: 8
```

**Latency-constrained deployment** (prefer simpler models; families are the lever, see [06 Deployment](06_deployment.md#choosing-a-model-that-fits-your-latency-budget)):

```yaml
search:
  families: [logreg, rf]
```

## Validation

Errors are raised for:

- a missing `target` (`config.target must be set`)
- an unknown key anywhere in the file
- a family name not in the registry (`Unknown model family`)
- a target column that is not in the data

Value ranges (for example `eta < 2`) are **not** validated yet. Keep values in the ranges above. Validation is on the [roadmap](09_roadmap.md).
