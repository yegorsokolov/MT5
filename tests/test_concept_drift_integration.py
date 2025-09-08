import importlib.util
import sys
import types
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "concept_drift",
    Path(__file__).resolve().parents[1] / "analysis" / "concept_drift.py",
)
concept_drift = importlib.util.module_from_spec(spec)
sys.modules["concept_drift"] = concept_drift
spec.loader.exec_module(concept_drift)


def test_concept_drift_integration(monkeypatch, tmp_path):
    logged = {}
    monkeypatch.setattr(
        concept_drift,
        "mlflow",
        types.SimpleNamespace(log_metric=lambda k, v: logged.setdefault(k, v)),
    )

    monitor = concept_drift.ConceptDriftMonitor(log_dir=tmp_path, delta=0.01)
    for _ in range(30):
        monitor.update({"f": 0.0}, prediction=0.0)

    triggered = False
    for _ in range(30):
        if monitor.update({"f": 1.0}, prediction=1.0):
            triggered = True
            break

    assert triggered, "Drift should be detected after distribution change"
    events = list(monitor.store.iter_events("retrain"))
    assert events, "Retrain event should be recorded"
    assert "concept_drift" in logged
