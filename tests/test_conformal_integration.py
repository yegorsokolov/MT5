import numpy as np
from pathlib import Path
import sys
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault(
    "telemetry",
    types.SimpleNamespace(
        get_meter=lambda name: types.SimpleNamespace(
            create_histogram=lambda *a, **k: types.SimpleNamespace(
                record=lambda *a, **k: None
            )
        )
    ),
)

sys.modules.setdefault(
    "services.message_bus",
    types.SimpleNamespace(Topics=types.SimpleNamespace(SIGNALS="signals"), get_message_bus=lambda *a, **k: None, MessageBus=object),
)
sys.modules.setdefault(
    "analysis.pipeline_anomaly",
    types.SimpleNamespace(validate=lambda df: True),
)
sys.modules.setdefault(
    "pandas",
    types.SimpleNamespace(DataFrame=object, Series=object),
)
event_writer_stub = types.SimpleNamespace(record=lambda *a, **k: None)
sys.modules.setdefault("event_store.event_writer", event_writer_stub)
sys.modules.setdefault("event_store", types.SimpleNamespace(event_writer=event_writer_stub))

from models import conformal
from mt5.signal_queue import _wrap_ci


def test_conformal_interval_coverage():
    rng = np.random.default_rng(1)
    X = rng.normal(size=2000)
    noise = rng.normal(scale=0.5, size=2000)
    y = X + noise
    preds_calib = X[:1000]
    y_calib = y[:1000]
    q = conformal.fit_residuals(y_calib - preds_calib, alpha=0.1)
    preds_test = X[1000:]
    y_test = y[1000:]
    lower, upper = conformal.predict_interval(preds_test, q)
    cov = conformal.evaluate_coverage(y_test, lower, upper)
    assert abs(cov - 0.9) < 0.05


def test_wrap_ci_computes_coverage():
    row = {"pred": 0.5, "ci_lower": 0.4, "ci_upper": 0.6, "y_true": 0.55}
    wrapped = _wrap_ci(row)
    assert wrapped["prediction"]["lower"] == 0.4
    assert wrapped["interval_covered"] == 1
