import sys
import numpy as np
import pandas as pd
import types
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
    "train_ensemble", Path(__file__).resolve().parents[1] / "mt5" / "train_ensemble.py"
)
_te = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["train_ensemble"] = _te
sys.modules["mt5.train_ensemble"] = _te
_spec.loader.exec_module(_te)  # type: ignore
main = _te.main


def _run(div_weight: bool) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    y = ((0.1 * X["a"] + X["b"]) > 0).astype(int)
    cfg = {
        "risk_per_trade": 0.1,
        "symbols": ["EURUSD"],
        "ensemble": {
            "enabled": True,
            "diversity_weighting": div_weight,
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
                    "features": ["a"],
                    "params": {
                        "n_estimators": 5,
                        "min_child_samples": 1,
                        "min_data_in_bin": 1,
                        "random_state": 0,
                    },
                },
                "m3": {
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
        metrics = main(cfg=cfg, data=(X, y))
    weights = logs.get("ensemble_weights.json", {})
    return metrics["ensemble"], weights


def test_diversity_weighting_shifts_weights_to_diverse_model():
    base_f1, base_w = _run(False)
    div_f1, div_w = _run(True)
    assert div_w["m3"] > base_w["m3"]
    assert div_f1 >= base_f1
