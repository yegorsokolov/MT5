import sys
import numpy as np
import pandas as pd
import types
import contextlib
import importlib.util
from pathlib import Path

# Ensure real scipy modules are available for sklearn
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
sys.modules.pop("scipy.sparse", None)
import scipy  # noqa: F401
import scipy.stats  # noqa: F401
import scipy.sparse  # noqa: F401

# Provide a minimal mlflow stub used by analytics.mlflow_client
mlflow_stub = types.SimpleNamespace(
    set_tracking_uri=lambda *a, **k: None,
    set_experiment=lambda *a, **k: None,
    start_run=lambda *a, **k: contextlib.nullcontext(),
    log_dict=lambda *a, **k: None,
    log_metric=lambda *a, **k: None,
    log_metrics=lambda *a, **k: None,
    end_run=lambda *a, **k: None,
)
sys.modules["mlflow"] = mlflow_stub

_spec = importlib.util.spec_from_file_location(
    "train_ensemble", Path(__file__).resolve().parents[1] / "train_ensemble.py"
)
_te = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["train_ensemble"] = _te
_spec.loader.exec_module(_te)  # type: ignore
main = _te.main


def test_ensemble_beats_base_models():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    y = ((X["a"] + X["b"]) > 0).astype(int)

    cfg = {
        "risk_per_trade": 0.1,
        "symbols": ["EURUSD"],
        "ensemble": {
            "enabled": True,
            "base_models": {
                "m1": {
                    "type": "lightgbm",
                    "features": ["a"],
                    "params": {
                        "n_estimators": 5,
                        "min_child_samples": 1,
                        "min_data_in_bin": 1,
                    },
                },
                "m2": {
                    "type": "lightgbm",
                    "features": ["b"],
                    "params": {
                        "n_estimators": 5,
                        "min_child_samples": 1,
                        "min_data_in_bin": 1,
                    },
                },
            },
            "meta_learner": True,
        },
    }

    metrics = main(cfg=cfg, data=(X, y))
    assert metrics["ensemble"] > max(metrics["m1"], metrics["m2"])
