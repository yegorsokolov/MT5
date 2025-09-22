import sys
import types
import numpy as np
import pandas as pd
import contextlib
import importlib.util
from pathlib import Path
from unittest import mock

# Ensure real scipy modules are available for sklearn
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
sys.modules.pop("scipy.sparse", None)
import scipy  # noqa: F401
import scipy.stats  # noqa: F401
import scipy.sparse  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "train_ensemble", ROOT / "mt5" / "train_ensemble.py"
)
_te = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["train_ensemble"] = _te
sys.modules["mt5.train_ensemble"] = _te
_spec.loader.exec_module(_te)  # type: ignore
main = _te.main
rng = np.random.default_rng(0)


def _run(use_ic: bool = True) -> tuple[dict[str, float], dict[str, dict]]:
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "b": -a + rng.normal(size=200)})
    y = (a + 0.5 * rng.normal(size=200) > 0).astype(int)
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
                        "random_state": 0,
                    },
                },
                "m2": {
                    "type": "lightgbm",
                    "features": ["b"],
                    "params": {
                        "n_estimators": 5,
                        "min_child_samples": 1,
                        "min_data_in_bin": 1,
                        "random_state": 0,
                    },
                },
            },
        },
    }
    logs: dict[str, dict] = {}
    mlflow_stub = types.SimpleNamespace(
        start_run=lambda *a, **k: contextlib.nullcontext(),
        log_dict=lambda data, name, *a, **k: logs.setdefault(name, data),
        log_metric=lambda *a, **k: None,
        log_metrics=lambda *a, **k: None,
        end_run=lambda *a, **k: None,
    )
    with mock.patch.dict(sys.modules, {"analytics.mlflow_client": mlflow_stub}):
        patcher = (
            contextlib.nullcontext()
            if use_ic
            else mock.patch(
                "train_ensemble.information_coefficient",
                return_value=float("nan"),
            )
        )
        with patcher:
            metrics = main(cfg=cfg, data=(X, y))
    return metrics, logs


def test_information_coefficient_weighting_improves_ensemble():
    metrics_w, logs_w = _run(True)
    coeffs = logs_w["information_coefficients.json"]
    weights = logs_w["ensemble_weights.json"]
    assert coeffs["m1"] > coeffs["m2"]
    assert weights["m1"] > weights["m2"]
    assert metrics_w["ensemble"] >= max(metrics_w["m1"], metrics_w["m2"])
    metrics_eq, logs_eq = _run(False)
    eq_w = logs_eq["ensemble_weights.json"]
    assert abs(eq_w["m1"] - eq_w["m2"]) < 1e-6
