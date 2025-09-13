import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

spec_cd = importlib.util.spec_from_file_location(
    "concept_drift",
    ROOT / "analysis" / "concept_drift.py",
)
concept_drift = importlib.util.module_from_spec(spec_cd)
sys.modules["concept_drift"] = concept_drift
spec_cd.loader.exec_module(concept_drift)

spec_dh = importlib.util.spec_from_file_location(
    "drift_handler", ROOT / "monitor" / "drift_handler.py"
)
drift_handler = importlib.util.module_from_spec(spec_dh)
sys.modules["drift_handler"] = drift_handler
spec_dh.loader.exec_module(drift_handler)


def test_drift_handler_triggers_retrain(tmp_path):
    calls = []

    def fake_retrain(model: str, **_: object) -> None:
        calls.append(model)

    monitor = concept_drift.ConceptDriftMonitor(log_dir=tmp_path, delta=0.01)
    handler = drift_handler.DriftHandler(
        monitor, threshold=1, cooldown=0.0, retrain_cb=fake_retrain
    )

    rng = np.random.default_rng(0)
    for _ in range(30):
        feats = {"f": rng.normal()}
        handler.update(feats, prediction=0.0, model="m")

    triggered = False
    for _ in range(30):
        feats = {"f": rng.normal(loc=5.0)}
        if handler.update(feats, prediction=1.0, model="m"):
            triggered = True
            break

    assert triggered, "Drift should be detected"
    assert calls, "Retrain should be triggered"
