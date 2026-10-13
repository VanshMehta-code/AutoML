"""Minimal prediction service. Run with:

    MODEL_PATH=artifacts/model.joblib uvicorn deploy.app:app --port 8000

See docs/06_deployment.md for the full runbook.
"""

import os

import pandas as pd
from fastapi import FastAPI, HTTPException

from automl.model import FittedModel

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
model = FittedModel.load(MODEL_PATH)

app = FastAPI(title="AutoML predictor", version="0.1.0")


@app.get("/health")
def health() -> dict:
    info = {"status": "ok", "task": model.task}
    if model.classes is not None:
        info["classes"] = [str(c) for c in model.classes]
    return info


@app.post("/predict")
def predict(records: list[dict]) -> dict:
    try:
        df = pd.DataFrame(records)
        if model.task == "regression":
            return {"predictions": model.predict(df).tolist()}
        return {"classes": [str(c) for c in model.classes],
                "probabilities": model.predict_proba(df).tolist()}
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
