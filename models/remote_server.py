"""Simple FastAPI server exposing heavy models for remote inference."""

from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Remote Model Server")
_MODEL_CACHE: Dict[str, object] = {}
_MODEL_DIR = Path(__file__).resolve().parent


class PredictRequest(BaseModel):
    model_name: str
    features: List[Dict[str, float]]


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, List[float]]:
    """Return predictions for ``req.features`` using ``req.model_name``."""
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
