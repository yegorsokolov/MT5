import json
import json
from pathlib import Path
import sys
import pathlib

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from analysis import decision_reviewer


class DummyLLM:
    def __call__(self, prompt: str) -> str:  # pragma: no cover - trivial
        return (
            "SUMMARY: ok\n"
            "FEATURE: add feature\n"
            "MODEL: deeper net\n"
            "MANUAL: 0\n"
            "RETRAIN: 1\n"
        )


def test_review_generates_report(tmp_path, monkeypatch):
    df = pd.DataFrame({"reason": ["bad", "worse"]})
    monkeypatch.setattr(decision_reviewer, "REVIEW_DIR", tmp_path)
    flags = decision_reviewer.review_rationales(
        llm=DummyLLM(), decisions=df, batch_size=2
    )
    files = list(Path(tmp_path).glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["feature_changes"] == ["add feature"]
    assert data["model_changes"] == ["deeper net"]
    assert data["flagged"] == {"manual": ["0"], "retrain": ["1"]}
    assert flags == {"manual": ["0"], "retrain": ["1"]}
