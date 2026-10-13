# 06 Deployment

This is the runbook for putting a trained model into production. It covers what gets deployed, how to serve it, how to check it works, and how to roll back.

## What you deploy

One file: `model.joblib`, a serialised `FittedModel`. It contains the cleaning rules, encodings, feature mask, ensemble members, and weights. Ship it together with `report.json`, which records how the model was built and how it scored.

Do not deploy the training code path to serve predictions. Only the `automl` package and the model file are needed for inference.

**Version pinning matters.** A joblib file depends on the library versions used to create it, mainly scikit-learn and numpy. Serve with the same versions recorded in your environment at training time. Save `pip freeze > requirements-train.txt` next to each artifact.

## Artifact layout (recommended)

```
artifacts/
└── 2026-10-12_churn_v3/
    ├── model.joblib          # the fitted model
    ├── report.json           # build summary and scores
    ├── requirements.txt      # exact versions used to train
    └── CHANGELOG.md          # what changed and why (one line per version)
```

Name artifacts by date, task and version so a rollback is a single path change.

## Serving

### Option A: HTTP service (default)

```bash
MODEL_PATH=artifacts/2026-10-12_churn_v3/model.joblib \
  uvicorn deploy.app:app --host 0.0.0.0 --port 8000 --workers 2
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check. Returns the class labels the model knows |
| POST | `/predict` | Body: a JSON list of records (objects with the input columns). Classification returns `probabilities` per class; regression returns `predictions` |

Input rows need the columns listed in `FittedModel.input_columns`. Extra columns are ignored. Missing columns return HTTP 422 with the names of the missing columns.

### Option B: batch file

```bash
automl predict --model artifacts/.../model.joblib --data new_customers.csv --out scores.csv
```

Use this for nightly scoring. It reads a CSV, writes a CSV of probabilities, and does not need a running server.

### Container (sketch)

A minimal Dockerfile looks like this. Adjust the base image and paths to your setup.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY deploy ./deploy
COPY artifacts/2026-10-12_churn_v3 ./model
RUN pip install --no-cache-dir ".[serve]"
ENV MODEL_PATH=/app/model/model.joblib
EXPOSE 8000
CMD ["uvicorn", "deploy.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Before you ship: checklist

- [ ] `report.json` shows `test_auc` within about 0.02 of `best_cv_auc`. A larger gap suggests overfitting to the search (see [07 Evaluation](07_evaluation.md#overfitting-to-the-search))
- [ ] `test_accuracy` is above the majority-class baseline (`min_class_frac` tells you the baseline in binary tasks: accuracy of always predicting the larger class is `1 − min_class_frac`)
- [ ] `pytest -q` passes on the exact commit that produced the artifact
- [ ] `/health` returns 200 and the expected class labels
- [ ] A smoke-test request with real sample rows returns probabilities that sum to 1
- [ ] Latency measured on the target hardware meets your budget (see below)
- [ ] The artifact and `requirements.txt` are stored together and versioned

## Latency and throughput

Tree ensembles are fast per row but slow to load if very large. Measure, do not guess:

```python
import time, pandas as pd
from automl.model import FittedModel
m = FittedModel.load("model.joblib")
rows = pd.read_csv("sample.csv").head(1000)
t = time.perf_counter(); m.predict_proba(rows); dt = time.perf_counter() - t
print(f"{dt / len(rows) * 1000:.3f} ms per row (batch of {len(rows)})")
```

Batch requests are much cheaper per row than single requests, because the preprocessing and model calls have fixed overhead. For real-time use, send single rows through the HTTP service and measure its end-to-end time, including the network.

### Choosing a model that fits your latency budget

The search optimises AUC only today. Latency is not yet part of the search objective (that is on the [roadmap](09_roadmap.md)). Until then, control it through the families you allow:

| Latency need | Suggested `families` | Notes |
|---|---|---|
| Under 1 ms per row, CPU | `[logreg]` | Fastest. Usually the lowest AUC on non-linear data |
| Low single-digit ms | `[hgb]` with a capped `max_iter` in the registry | Fast boosted trees |
| Can afford more | `[hgb, rf]` | `rf` with many trees is slower. Lower `n_estimators` in `models.py` to speed it up |

Ensembles multiply the cost: every member runs on every request. If latency is tight, set `ensemble_size: 1`.

## Monitoring in production

Track these, and alert on the first two:

1. **Input drift.** Compare the mean and missing-rate of each input column against the training metadata. `report.json` does not yet store per-column baselines, so save `metadata` from the training run (it is available on `model.preprocessor.metadata`) and compare against it. A quick check: column missing rate has moved by more than 5 points, or a numeric column's mean has moved by more than 3 training standard deviations.
2. **Unseen categories.** A rise in unseen levels means the world has changed. The model silently treats them as zero-vector one-hot, so monitor the count yourself.
3. **Prediction distribution.** The share of predicted positives (or the mean predicted probability) should stay near its test-time value. A sharp shift is a warning, even without labels.
4. **Outcome quality.** When labels arrive, compute AUC on recent predictions. This is the only way to see true performance decay.

Monitoring tooling is not built yet. See the roadmap.

## Rollback

1. Keep the previous artifact directory in place. Do not overwrite it.
2. Point `MODEL_PATH` at the previous `model.joblib` and restart the service.
3. Verify with `/health` and a smoke-test request.
4. Record the reason in `CHANGELOG.md`.

Rollback is a configuration change, not a retrain, so it takes seconds.

## Retraining

Retrain when monitoring shows drift or quality decay, or on a fixed schedule (for example monthly). Each retrain produces a new dated artifact, which goes through the checklist above before it replaces the live one. Never retrain and deploy in one step without the test-AUC check.

## Known deployment limits

- The model is **not** calibrated. Probabilities are ranking scores. If you need real probabilities (for example to set a threshold on expected cost), calibrate on held-out data first. This is planned, see the roadmap.
- The service loads one model per process. Scale with workers or replicas.
- There is no authentication in `deploy/app.py`. Put it behind your gateway or add an API key before exposing it.
