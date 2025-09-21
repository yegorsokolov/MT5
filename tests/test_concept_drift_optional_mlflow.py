import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_concept_drift_without_mlflow(monkeypatch, tmp_path):
    """Concept drift monitor should operate when MLflow is missing."""

    monkeypatch.delitem(sys.modules, "analysis.concept_drift", raising=False)
    monkeypatch.delitem(sys.modules, "analytics.mlflow_client", raising=False)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    scheduler_stub = types.ModuleType("scheduler")

    def _schedule_retrain(*, model: str = "classic", update_hyperparams: bool = False, store=None):
        if store is not None:
            payload = {"model": model}
            if update_hyperparams:
                payload["update_hyperparams"] = True
            store.record("retrain", payload)

    scheduler_stub.schedule_retrain = _schedule_retrain  # type: ignore[attr-defined]
    scheduler_stub.__spec__ = importlib.machinery.ModuleSpec("scheduler", loader=None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scheduler", scheduler_stub)

    concept_drift = importlib.import_module("analysis.concept_drift")
    assert not concept_drift._MLFLOW_AVAILABLE

    pandas_stub = sys.modules.get("pandas")
    if pandas_stub is not None:
        if not hasattr(pandas_stub, "Series"):
            pandas_stub.Series = tuple  # type: ignore[attr-defined]

        if not hasattr(pandas_stub, "Timestamp"):
            class _Timestamp:
                @staticmethod
                def utcnow():  # pragma: no cover - simple stub
                    class _Now:
                        @staticmethod
                        def isoformat() -> str:
                            return "1970-01-01T00:00:00"

                    return _Now()

            pandas_stub.Timestamp = _Timestamp  # type: ignore[attr-defined]

    monitor = concept_drift.ConceptDriftMonitor(log_dir=tmp_path)

    class DriftOnce:
        def __init__(self, trigger: bool = True) -> None:
            self.trigger = trigger

        def update(self, value: float) -> bool:
            if self.trigger:
                self.trigger = False
                return True
            return False

    monitor._feat_detectors["feature"] = DriftOnce()
    monitor._pred_detector = DriftOnce(trigger=False)

    assert monitor.update({"feature": 1.0}, prediction=0.5)
    events = list(monitor.store.iter_events("retrain"))
    assert events, "Drift events should be recorded even without MLflow"
