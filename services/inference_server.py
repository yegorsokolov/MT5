from __future__ import annotations

"""Unified inference server exposing models over HTTP and gRPC.

The server is intentionally lightweight – it focuses on providing an
interface that can host multiple models with optional GPU acceleration and
basic autoscaling hooks.  Heavy lifting such as actual autoscaling
infrastructure is expected to be handled by the deployment environment (e.g.
Kubernetes or Ray).  For the purposes of the tests this module simply loads
models on demand and serves predictions.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import asyncio
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

app = FastAPI(title="Inference Server")

# Cache of loaded models keyed by model name
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# Simple thread pool to provide a bit of concurrency.  The default can be
# overridden via the ``INFER_WORKERS`` environment variable which allows the
# hosting platform to scale according to available resources.
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("INFER_WORKERS", "4")))


class PredictRequest(BaseModel):
    """Request payload for prediction calls."""

    model_name: str
    features: List[Dict[str, float]]


@app.get("/health")
def health() -> Dict[str, Any]:
    """Basic health check used by the client for liveness probes."""

    return {"status": "ok", "loaded_models": list(_MODEL_CACHE.keys())}


def _load_model(name: str) -> Any:
    """Load ``name`` from disk if not already cached."""

    model = _MODEL_CACHE.get(name)
    if model is None:
        path = _MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            raise HTTPException(status_code=404, detail="model not found")
        model = joblib.load(path)
        # If a GPU is available move the model there.  This is a no-op for
        # models that do not implement ``to``.
        if torch and torch.cuda.is_available() and hasattr(model, "to"):
            try:
                model.to("cuda")
            except Exception:  # pragma: no cover - defensive
                pass
        _MODEL_CACHE[name] = model
    return model


@app.post("/predict")
async def predict(req: PredictRequest) -> Dict[str, List[float]]:
    """Return predictions for ``req.features`` using ``req.model_name``."""

    model = _load_model(req.model_name)
    df = pd.DataFrame(req.features)
    loop = asyncio.get_event_loop()

    def _do_predict() -> List[float]:
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(df)[:, 1]
        elif hasattr(model, "predict"):
            preds = model.predict(df)
        else:  # pragma: no cover - defensive programming
            raise HTTPException(status_code=400, detail="model lacks prediction API")
        if hasattr(preds, "tolist"):
            return preds.tolist()
        return list(preds)

    preds = await loop.run_in_executor(_EXECUTOR, _do_predict)
    return {"predictions": preds}


# ---------------------------------------------------------------------------
# gRPC support
# ---------------------------------------------------------------------------
# To keep the code lightweight we avoid defining dedicated protobuf messages
# and instead rely on ``google.protobuf.struct_pb2.Struct`` which behaves like a
# JSON object.  The client sends a Struct with ``model_name`` and ``features``
# fields mirroring :class:`PredictRequest` above and receives a Struct with a
# ``predictions`` field.
try:  # pragma: no cover - optional runtime dependency
    import grpc
    from google.protobuf import struct_pb2

    class _GrpcHandler(grpc.aio.GenericRpcHandler):
        def service_name(self) -> str:
            return "inference.InferenceService"

        def unary_unary(self, method: str, request_deserializer, response_serializer):
            if method == "/inference.InferenceService/Predict":
                async def handler(request: struct_pb2.Struct, context):
                    payload = PredictRequest(
                        model_name=request["model_name"],  # type: ignore[index]
                        features=[dict(x.fields) for x in request["features"]],  # type: ignore[index]
                    )
                    result = await predict(payload)
                    resp = struct_pb2.Struct()
                    resp.update(result)
                    return resp

                return grpc.aio.unary_unary_rpc_method_handler(
                    handler,
                    request_deserializer=struct_pb2.Struct.FromString,
                    response_serializer=struct_pb2.Struct.SerializeToString,
                )
            if method == "/inference.InferenceService/Health":
                async def handler(request: struct_pb2.Struct, context):  # type: ignore[misc]
                    resp = struct_pb2.Struct()
                    resp.update(health())
                    return resp

                return grpc.aio.unary_unary_rpc_method_handler(
                    handler,
                    request_deserializer=struct_pb2.Struct.FromString,
                    response_serializer=struct_pb2.Struct.SerializeToString,
                )
            return None

    async def serve_grpc(address: str = "[::]:8500") -> None:
        """Run a lightweight gRPC server in parallel to the HTTP API."""

        server = grpc.aio.server()
        server.add_generic_rpc_handlers((_GrpcHandler(),))
        server.add_insecure_port(address)
        await server.start()
        await server.wait_for_termination()

except Exception:  # pragma: no cover - gRPC optional
    grpc = None  # type: ignore
    serve_grpc = None  # type: ignore


if __name__ == "__main__":  # pragma: no cover - service entry point
    import uvicorn

    # Optionally start the gRPC server alongside the FastAPI application
    if serve_grpc is not None:
        loop = asyncio.get_event_loop()
        loop.create_task(serve_grpc())
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
