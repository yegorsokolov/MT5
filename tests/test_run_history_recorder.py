from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reports.run_history import RunHistoryRecorder


def test_run_history_recorder_copies_artifacts(tmp_path):
    repo_root = tmp_path
    (repo_root / ".git").mkdir()

    log_path = repo_root / "app.log"
    log_path.write_text("line-0\n" + "line-1\n" * 50)

    recorder = RunHistoryRecorder(
        component="unit-test",
        config={"foo": "bar"},
        tags={"mode": "test"},
        repo_root=repo_root,
    )
    recorder.add_artifact(log_path, dest_name="logs/app.log", max_bytes=40)
    recorder.start()
    recorder.set_metrics({"score": 0.75})
    recorder.update_context(note="demo")
    recorder.finish(status="completed", result=0.75)

    history_root = repo_root / "reports" / "run_history"
    run_dir = history_root / recorder.run_id
    record_path = run_dir / "run.json"
    assert record_path.exists()

    payload = json.loads(record_path.read_text())
    assert payload["metrics"]["score"] == 0.75
    assert payload["status"] == "completed"
    assert payload["context"]["note"] == "demo"

    artifact = payload["artifacts"][0]
    assert artifact["copied"] is True
    copied_path = Path(artifact["destination"])
    # Destination is stored relative to repo root.
    copied_file = repo_root / copied_path
    assert copied_file.exists()
    truncated_content = copied_file.read_text()
    assert "truncated" in truncated_content

    index_path = history_root / "index.json"
    index_payload = json.loads(index_path.read_text())
    assert index_payload["runs"][0]["run_id"] == recorder.run_id

    latest_path = history_root / "latest.json"
    latest_payload = json.loads(latest_path.read_text())
    assert latest_payload["run_id"] == recorder.run_id
