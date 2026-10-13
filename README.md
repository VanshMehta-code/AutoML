# AutoML

A deployment-oriented AutoML system for tabular classification. It cleans data using dataset metadata, builds a feature graph, selects features with ant colony optimisation, searches models with an evolutionary algorithm, and ships one artifact that serves predictions.

**Full documentation is in [`docs/`](docs/README.md).** Start with the [quick start](docs/01_quickstart.md).

## Quick commands

```bash
pip install -e ".[dev,serve]"                                  # install
pytest -q                                                      # run tests
automl train --data data/train.csv --target y --out artifacts/run1
automl predict --model artifacts/run1/model.joblib --data new.csv --out preds.csv
MODEL_PATH=artifacts/run1/model.joblib uvicorn deploy.app:app  # serve over HTTP
```

## Project layout

```
automl-project/
├── configs/default.yaml    # all tunable settings
├── docs/                   # documentation (start at docs/README.md)
├── src/automl/             # the library
├── deploy/app.py           # HTTP prediction service
└── tests/                  # test suite
```

## Status

Binary and multiclass classification are implemented and tested, with end-to-end tests for both. Regression, gradient-boosting libraries, feature generation with genetic programming, latency-aware search, calibration and drift monitoring are planned. The status table is in [docs/09_roadmap.md](docs/09_roadmap.md).

## Contributing

1. Read [docs/02_architecture.md](docs/02_architecture.md) to find the right module.
2. Add or change behaviour, and add a test that would fail without it.
3. Update the affected doc in `docs/` in the same change. A doc that disagrees with the code is a bug.
4. Record any significant design choice in [docs/08_decisions.md](docs/08_decisions.md).
5. Keep code comments short. Put explanations in `docs/`.
