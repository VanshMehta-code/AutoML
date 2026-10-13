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

app = FastAPI(title="AutoML predictor", version=model.report.get("task", "unknown"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classes": [str(c) for c in model.classes]}


@app.post("/predict")
def predict(records: list[dict]) -> dict:
    try:
        proba = model.predict_proba(pd.DataFrame(records))
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    return {"classes": [str(c) for c in model.classes], "probabilities": proba.tolist()}
