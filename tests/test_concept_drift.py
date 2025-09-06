import importlib.util
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

spec = importlib.util.spec_from_file_location(
    "concept_drift", Path(__file__).resolve().parents[1] / "analysis" / "concept_drift.py"
)
concept_drift = importlib.util.module_from_spec(spec)
sys.modules["concept_drift"] = concept_drift
spec.loader.exec_module(concept_drift)

def test_concept_drift_detection(tmp_path):
    log_dir = tmp_path / "logs"
    monitor = concept_drift.ConceptDriftMonitor(log_dir=log_dir, delta=0.01)

    for _ in range(30):
        monitor.update({"f": 0.0}, prediction=0.0)

    triggered = False
    for _ in range(30):
        triggered = monitor.update({"f": 1.0}, prediction=1.0)
        if triggered:
            break

    assert triggered, "Drift should be detected after distribution change"
    events = list(monitor.store.iter_events("retrain"))
    assert events and events[0]["payload"].get("update_hyperparams")
    assert any(log_dir.iterdir()), "Drift log file should be created"
