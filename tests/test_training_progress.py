import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from training.progress import TrainingProgressTracker


def test_training_progress_tracker_writes_and_completes(tmp_path: Path) -> None:
    out_path = tmp_path / "progress.json"
    tracker = TrainingProgressTracker(output_path=out_path, total_steps=2)

    payload = json.loads(out_path.read_text())
    assert payload["status"] == "idle"
    assert payload["step"] == 0

    tracker.start("initialising")
    payload = json.loads(out_path.read_text())
    assert payload["status"] == "running"
    assert payload["stage"] == "initialising"

    tracker.advance("history_loaded")
    payload = json.loads(out_path.read_text())
    assert payload["step"] == 1
    assert payload["stage"] == "history_loaded"

    tracker.complete(runtime_seconds=3.5)
    payload = json.loads(out_path.read_text())
    assert payload["status"] == "completed"
    assert payload["stage"] == "completed"
    assert payload["runtime_seconds"] == pytest.approx(3.5)


def test_training_progress_tracker_fail(tmp_path: Path) -> None:
    out_path = tmp_path / "progress.json"
    tracker = TrainingProgressTracker(output_path=out_path, total_steps=1)
    tracker.start("initialising")
    tracker.fail("loading_history", "boom", runtime_seconds=1.2)

    payload = json.loads(out_path.read_text())
    assert payload["status"] == "failed"
    assert payload["stage"] == "loading_history"
    assert payload["runtime_seconds"] == pytest.approx(1.2)
    assert payload["error"] == "boom"


def test_training_progress_tracker_advance_without_explicit_start(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "progress.json"
    tracker = TrainingProgressTracker(output_path=out_path, total_steps=2)

    tracker.advance("warmup")
    payload = json.loads(out_path.read_text())
    assert payload["status"] == "running"
    assert payload["stage"] == "warmup"
    assert payload["step"] == 1
    assert payload["total_steps"] == 2

    tracker.advance("training")
    tracker.advance("extra_step")
    payload = json.loads(out_path.read_text())
    assert payload["stage"] == "extra_step"
    assert payload["step"] == 2, "step should saturate at total_steps"
