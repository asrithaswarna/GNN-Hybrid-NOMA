"""Optional FastAPI service for the classical rate model artifact."""

from pathlib import Path
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import RatePredictionRequest, RatePredictionResponse
from src.classical_models import predict_rate
from src.monitoring import PredictionMonitor

app = FastAPI(title="Version C Rate Prediction API")
MODEL_PATH = Path("results/classical_models/random_forest.joblib")
monitor = PredictionMonitor("results/monitoring/predictions.json")


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info():
    return {"model": "Random Forest", "artifact": str(MODEL_PATH), "task": "user rate regression"}


@app.post("/predict", response_model=RatePredictionResponse)
def predict(request: RatePredictionRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact is not available; run compare first")
    frame = pd.DataFrame([request.model_dump()])
    frame["channel_gain_db"] = 10.0 * __import__("numpy").log10(max(request.channel_gain, 1e-30))
    frame["snr_db"] = 0.0
    start = perf_counter()
    try:
        value = float(predict_rate(str(MODEL_PATH), frame)[0])
    except ValueError as exc:
        monitor.record(None, (perf_counter() - start) * 1000.0, valid=False)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    latency = (perf_counter() - start) * 1000.0
    monitor.record(value, latency)
    return RatePredictionResponse(predicted_rate_mbps=value, model="Random Forest")
