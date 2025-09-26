from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt5.run_state import PipelineState


def test_pipeline_state_resume_flow(tmp_path):
    (tmp_path / ".git").mkdir()
    state_path = tmp_path / "state.json"
    state = PipelineState(state_path, repo_root=tmp_path)

    state.begin_run(args={}, resume_from=None)
    assert state.resume_stage() == "training"

    artifact = tmp_path / "model.joblib"
    artifact.write_text("ok")
    state.mark_stage_complete("training", artifacts=[artifact])
    state.mark_run_failed("interrupted")

    assert state.should_resume() is True
    assert state.resume_stage() == "backtest"

    artifact.unlink()
    assert state.resume_stage() == "training"

    state.reset()
    assert state.resume_stage() == "training"
