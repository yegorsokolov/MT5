"""Simple FastAPI server exposing heavy models for remote inference."""

from pathlib import Path
from typing import Deque, Dict, List
import time
from collections import deque

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:  # pragma: no cover - torch optional
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

app = FastAPI(title="Remote Model Server")
_MODEL_CACHE: Dict[str, object] = {}
_MODEL_DIR = Path(__file__).resolve().parent
# Track recent request timestamps for autoscaling metrics
_REQUEST_TIMES: Deque[float] = deque()


class PredictRequest(BaseModel):
    model_name: str
    features: List[Dict[str, float]]


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, List[float]]:
    """Return predictions for ``req.features`` using ``req.model_name``."""
    record_request()
    model = _MODEL_CACHE.get(req.model_name)
    if model is None:
        path = _MODEL_DIR / f"{req.model_name}.pkl"
        if not path.exists():
            raise HTTPException(status_code=404, detail="model not found")
        model = joblib.load(path)
        _MODEL_CACHE[req.model_name] = model
    df = pd.DataFrame(req.features)
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(df)[:, 1]
    elif hasattr(model, "predict"):
        preds = model.predict(df)
    else:  # pragma: no cover - defensive programming
        raise HTTPException(status_code=400, detail="model lacks prediction API")
    return {"predictions": preds.tolist()}


def record_request() -> None:
    """Record the timestamp of an incoming request."""

    now = time.time()
    _REQUEST_TIMES.append(now)
    cutoff = now - 10.0
    while _REQUEST_TIMES and _REQUEST_TIMES[0] < cutoff:
        _REQUEST_TIMES.popleft()


def get_request_rate(window: float = 10.0) -> float:
    """Return recent request rate in requests per second."""

    now = time.time()
    cutoff = now - window
    count = sum(1 for ts in _REQUEST_TIMES if ts >= cutoff)
    return count / window


def get_gpu_utilization() -> float:
    """Return current GPU memory utilisation ratio.

    When a GPU is unavailable this returns ``0.0``.  The implementation is
    intentionally lightweight; in real deployments a proper GPU monitoring
    library would be used.
    """

    if torch and torch.cuda.is_available():  # pragma: no cover - optional
        try:
            props = torch.cuda.get_device_properties(0)
            util = torch.cuda.memory_reserved(0) / float(props.total_memory)
            return float(util)
        except Exception:
            return 0.0
    return 0.0


__all__ = [
    "app",
    "PredictRequest",
    "predict",
    "record_request",
    "get_request_rate",
    "get_gpu_utilization",
]
