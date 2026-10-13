# 01 Quick start

This gets you from zero to a trained, saved, and served model.

## 1. Requirements

- Python 3.10 or newer
- About 2 GB of RAM for datasets up to roughly 100k rows (more for wider data)

## 2. Install

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,serve]"      # core + tests + FastAPI service
```

Check that it works:

```bash
automl --help
pytest -q                          # expect all tests to pass
```

## 3. Prepare your data

You need one CSV file with a header row and one target column.

- Classification only (binary or multiclass). Regression is not supported yet.
- Missing values are fine. Leave them blank.
- Keep the ID and free-text columns. The pipeline detects and handles them (see [Stages](03_stages.md#stage-3-cleaning-and-encoding)).

## 4. Set the target

Either set it in `configs/default.yaml`:

```yaml
target: churned
```

or pass it on the command line with `--target`, which overrides the file.

## 5. Train

```bash
automl train --data data/train.csv --target churned --out artifacts/run1
```

You will see progress like this:

```
[metadata] rows=1200 features=8 task=binary classes=2
[graph] encoded=13 redundant_dropped=1
[aco] selected 2/12 features
[evolve] gen 1/4  screened=16  full=5  best=logreg cv_auc=0.9442
[done] cv_auc=0.9442  test_auc=0.9227  test_acc=0.8333  members=1
```

The output folder contains:

- `model.joblib`: the complete fitted model (cleaning, features, ensemble)
- `report.json`: how it was built and how it scored (see [Evaluation](07_evaluation.md#reading-the-run-report))

Training time is capped by `search.time_budget_seconds` (default 900 s). Lower it for quick experiments.

## 6. Predict from the command line

```bash
automl predict --model artifacts/run1/model.joblib --data data/new_rows.csv --out preds.csv
```

For classification the output has one column per class with its probability. For regression it has one `prediction` column. The target column is optional in the input file and is ignored if present.

If the target is a continuous number, the task is detected as regression automatically. Force it with `--task regression` if needed.

Leaky columns are dropped automatically. The training log shows what was removed and why. To see the effect of a leak, compare a run with `--keep-leaky`.

## 7. Serve predictions over HTTP

```bash
MODEL_PATH=artifacts/run1/model.joblib uvicorn deploy.app:app --port 8000
```

Then:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '[{"x1": 0.4, "cat": "c3", "when": "2021-05-01", "x2": 1.2, "x2_copy": 1.2, "small_cat": "a", "uid": "u9", "const": 1}]'
```

Full deployment guidance is in [06 Deployment](06_deployment.md).

## 8. Next steps

- Tune the search budget and families: [05 Configuration](05_configuration.md)
- Understand what the numbers mean before trusting them: [07 Evaluation](07_evaluation.md)
- If something fails: [11 Troubleshooting](11_troubleshooting.md)
