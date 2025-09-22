import json
from pathlib import Path
import runpy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ingest = runpy.run_path(str(DATA_DIR / "ingest.py"), run_name="__test__")["ingest"]

VERSIONS_PATH = Path(__file__).resolve().parents[1] / "data_versions.json"


def _read_entry(path: Path) -> dict:
    data = json.loads(VERSIONS_PATH.read_text())
    return data["lineage"][str(path.resolve())]


def test_data_lineage_hash_changes(tmp_path):
    original = json.loads(VERSIONS_PATH.read_text())
    try:
        raw = tmp_path / "raw.csv"
        raw.write_text("a,b\n1,2\n")
        ingest(raw, record=True)
        first = _read_entry(raw)

        raw.write_text("a,b\n1,3\n")
        ingest(raw, record=True)
        second = _read_entry(raw)

        assert first["raw"] != second["raw"]
        assert first["features"] != second["features"]
    finally:
        VERSIONS_PATH.write_text(json.dumps(original, indent=2, sort_keys=True))


def test_data_lineage_env_fallback(monkeypatch, tmp_path):
    from analysis import data_lineage

    store_path = tmp_path / "lineage.parquet"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(data_lineage, "STORE_PATH", store_path)
    monkeypatch.setenv("MT5_RUN_ID", "env-run")

    data_lineage.log_lineage("unknown", "raw.csv", "transform", "feature")
    df = data_lineage.get_lineage("env-run")
    assert not df.empty
    assert df.iloc[0]["run_id"] == "env-run"
