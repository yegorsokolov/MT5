import subprocess
from pathlib import Path


def test_docs_doctest():
    build_dir = Path("docs/_build/doctest")
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["sphinx-build", "-b", "doctest", "-W", "docs", str(build_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
