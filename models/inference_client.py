"""Client for :mod:`services.inference_server`.

The client communicates with the inference server over HTTP or gRPC and
includes basic health checking and retry logic.  For the purposes of the
unit tests only the HTTP path is exercised, but the gRPC implementation is
provided for completeness.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Mapping, Sequence

import pandas as pd
import requests

try:  # pragma: no cover - optional dependency
    import grpc
    from google.protobuf import struct_pb2
except Exception:  # pragma: no cover
    grpc = None  # type: ignore
    struct_pb2 = None  # type: ignore


class InferenceClient:
    """Simple inference client with retry and health checking."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        retries: int = 2,
        backoff: float = 0.2,
    ) -> None:
        self.base_url = base_url or os.getenv("INFERENCE_SERVER_URL", "http://localhost:8000")
        self.retries = retries
        self.backoff = backoff

    # ------------------------------------------------------------------
    def health(self) -> bool:
        """Return ``True`` if the server responds successfully."""

        if self.base_url.startswith("grpc://") and grpc is not None:
            channel = grpc.insecure_channel(self.base_url[7:])
            stub = channel.unary_unary(
                "/inference.InferenceService/Health",
                request_serializer=struct_pb2.Struct.SerializeToString,
                response_deserializer=struct_pb2.Struct.FromString,
            )
            try:
                stub(struct_pb2.Struct(), timeout=5)
                return True
            except Exception:
                return False
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    def _to_records(self, features: Any) -> List[Mapping[str, Any]]:
        if isinstance(features, pd.DataFrame):
            return features.to_dict(orient="records")
        if isinstance(features, Sequence):
            return list(features)  # type: ignore[arg-type]
        raise TypeError(f"Unsupported features type: {type(features)!r}")

    # ------------------------------------------------------------------
    def predict(self, model_name: str, features: Any) -> List[float]:
        payload = {"model_name": model_name, "features": self._to_records(features)}
        attempt = 0
        while True:
            try:
                if self.base_url.startswith("grpc://") and grpc is not None:
                    channel = grpc.insecure_channel(self.base_url[7:])
                    stub = channel.unary_unary(
                        "/inference.InferenceService/Predict",
                        request_serializer=struct_pb2.Struct.SerializeToString,
                        response_deserializer=struct_pb2.Struct.FromString,
                    )
                    req = struct_pb2.Struct()
                    req.update(payload)
                    resp = stub(req, timeout=10)
                    preds = list(resp["predictions"])  # type: ignore[index]
                    return preds
                resp = requests.post(
                    f"{self.base_url}/predict", json=payload, timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                preds = data.get("predictions", [])
                if not isinstance(preds, list):
                    raise ValueError("Invalid response from inference server")
                return preds
            except Exception:
                attempt += 1
                if attempt > self.retries:
                    raise
                time.sleep(self.backoff * attempt)
