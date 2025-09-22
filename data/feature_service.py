"""Lightweight feature retrieval service.

The service exposes pre-computed features via a simple REST API. Clients send
requests over HTTPS including an API key. When features for a given symbol and
`start`/`end` range are missing, callers are expected to compute them locally and
upload them back to the service using the provided endpoint.

TLS certificates and API keys are supplied via the environment variables
`FEATURE_SERVICE_CERT`, `FEATURE_SERVICE_KEY` and `FEATURE_SERVICE_API_KEY`.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader

from .feature_store import FeatureStore
from utils.secret_manager import SecretManager

API_KEY_NAME = "X-API-Key"
API_KEY = SecretManager().get_secret("FEATURE_SERVICE_API_KEY") or ""
if not API_KEY:
    raise RuntimeError("FEATURE_SERVICE_API_KEY secret is required")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_key(api_key: str = Depends(api_key_header)) -> str:
    if API_KEY and api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI()


@app.get("/features/{symbol}")
async def get_features(
    symbol: str,
    start: str,
    end: str,
    api_key: str = Depends(verify_key),
) -> List[Dict[str, Any]]:
    """Return features for *symbol* between *start* and *end*.

    The response is a list of dictionaries suitable for constructing a
    ``pandas.DataFrame``. A 404 is raised if the features are not present in the
    underlying :class:`FeatureStore`.
    """

    store = FeatureStore()
    params = {"start": start, "end": end}
    df = store.load_any(symbol, 0, params)
    if df is None:
        raise HTTPException(status_code=404, detail="Features not found")
    return df.to_dict(orient="records")


@app.post("/features/{symbol}")
async def put_features(
    symbol: str,
    start: str,
    end: str,
    data: List[Dict[str, Any]],
    api_key: str = Depends(verify_key),
) -> Dict[str, str]:
    """Upload features for *symbol* and persist them in the store."""

    df = pd.DataFrame(data)
    store = FeatureStore()
    params = {"start": start, "end": end}
    store.save(df, symbol, 0, params, raw_hash="remote")
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover - service entrypoint
    import uvicorn

    cert = os.getenv("FEATURE_SERVICE_CERT")
    key = os.getenv("FEATURE_SERVICE_KEY")
    if not cert or not key:
        raise RuntimeError("TLS certificates must be provided via FEATURE_SERVICE_CERT and FEATURE_SERVICE_KEY")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("FEATURE_SERVICE_PORT", "8000")),
        ssl_certfile=cert,
        ssl_keyfile=key,
    )
