"""Client for the lightweight :mod:`services.inference_server`.

This module provides a minimal helper used by :class:`model_registry.ModelRegistry`
to offload predictions to a remote inference server when the local machine lacks
the required resources for a full sized model.  The interface intentionally
mimics that of a local model's ``predict`` method so callers can swap between
local and remote execution transparently.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable, List, Mapping, Sequence

try:  # pragma: no cover - pandas is optional for the tests
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

import requests

from services.worker_manager import get_worker_manager

# Base URL of the remote inference service.  Only the ``/predict`` endpoint is
# used by the tests but a ``/health`` check is also available on the server.
_BASE_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:8000")


def _to_records(features: Any) -> List[Mapping[str, Any]]:
    """Return *features* as a list of records suitable for JSON encoding.

    ``features`` may be a pandas ``DataFrame`` or any iterable of mappings.
    The pandas dependency is optional; if unavailable the function falls back
    to duck-typing via ``to_dict``.
    """

    df_type = getattr(pd, "DataFrame", None)
    try:
        if df_type is not None and isinstance(features, df_type):  # type: ignore[arg-type]
            return features.to_dict(orient="records")
    except TypeError:  # pandas may be stubbed with a callable
        pass
    if hasattr(features, "to_dict"):
        return features.to_dict(orient="records")  # type: ignore[call-arg]
    if isinstance(features, Sequence):
        return list(features)  # type: ignore[arg-type]
    raise TypeError(f"Unsupported features type: {type(features)!r}")


def predict(model_name: str, features: Any, batch_size: int | None = None) -> List[float]:
    """Return predictions for ``features`` from a remote model.

    Parameters
    ----------
    model_name:
        Identifier of the model on the remote server.
    features:
        Feature matrix as a pandas ``DataFrame`` or an iterable of dicts.
    batch_size:
        If provided, features are split into batches of this size and sent to
        the remote server sequentially.  This allows callers to trade off
        latency for throughput when dealing with large inputs.
    """

    records = _to_records(features)
    if not records:
        return []
    bs = max(1, batch_size or len(records))
    predictions: List[float] = []
    for i in range(0, len(records), bs):
        batch = records[i : i + bs]
        start = time.perf_counter()
        try:
            payload = {"model_name": model_name, "features": batch}
            resp = requests.post(f"{_BASE_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            preds = data.get("predictions", [])
            if not isinstance(preds, list):
                raise ValueError("Invalid response from remote model")
            predictions.extend(preds)
        finally:
            latency = time.perf_counter() - start
            get_worker_manager().record_request(
                "remote_client", latency, batch_size=len(batch)
            )
    return predictions


# Backwards compatibility for older code paths
predict_remote = predict


__all__ = ["predict", "predict_remote"]
