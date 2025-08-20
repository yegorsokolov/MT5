import os
import time
from typing import Any, Iterable, List, Mapping, Sequence

import pandas as pd
import requests

from services.worker_manager import get_worker_manager

REMOTE_URL = os.getenv("REMOTE_MODEL_URL", "http://localhost:8000/predict")


def _to_records(features: Any) -> List[Mapping[str, Any]]:
    """Return *features* as a list of records suitable for JSON encoding.

    ``features`` may be a pandas ``DataFrame`` or any iterable of mappings.
    """
    if isinstance(features, pd.DataFrame):
        return features.to_dict(orient="records")
    if isinstance(features, Sequence):
        return list(features)  # type: ignore[arg-type]
    raise TypeError("Unsupported features type: %r" % (type(features),))


def predict_remote(model_name: str, features: Any) -> List[float]:
    """Return predictions from a remote model via REST.

    The call latency is reported to the :class:`services.worker_manager.WorkerManager`
    so the system can scale worker processes based on demand.

    Parameters
    ----------
    model_name:
        Identifier of the model on the remote server.
    features:
        Feature matrix as a pandas ``DataFrame`` or an iterable of dicts.
    """
    start = time.perf_counter()
    try:
        payload = {"model_name": model_name, "features": _to_records(features)}
        resp = requests.post(REMOTE_URL, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        preds = data.get("predictions", [])
        if not isinstance(preds, list):
            raise ValueError("Invalid response from remote model")
        return preds
    finally:
        latency = time.perf_counter() - start
        get_worker_manager().record_request("remote_client", latency)
