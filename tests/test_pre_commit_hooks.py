from pathlib import Path
import subprocess


def test_pre_commit_rejects_bad_code(tmp_path):
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("x=1")
    result = subprocess.run(
        ["pre-commit", "run", "--files", str(bad_file)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
