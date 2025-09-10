import importlib.util
from pathlib import Path
import subprocess
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
spec = importlib.util.spec_from_file_location(
    "model_store", ROOT / "models" / "model_store.py"
)
model_store = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(model_store)  # type: ignore


def test_save_and_load(tmp_path):
    model = {"weights": [1, 2, 3]}
    cfg = {"lr": 0.1}
    perf = {"accuracy": 0.9}
    version_id = model_store.save_model(model, cfg, perf, store_dir=tmp_path)
    loaded, meta = model_store.load_model(version_id, store_dir=tmp_path)
    assert loaded == model
    assert meta["training_config"] == cfg
    assert meta["performance"] == perf


def test_list_versions(tmp_path):
    model_store.save_model({"a": 1}, {}, {}, store_dir=tmp_path)
    versions = model_store.list_versions(store_dir=tmp_path)
    assert len(versions) == 1
    assert "version_id" in versions[0]


def test_provenance_metadata(tmp_path):
    model = {"weights": [1]}
    version_id = model_store.save_model(model, {}, {}, store_dir=tmp_path)
    _, meta = model_store.load_model(version_id, store_dir=tmp_path)
    # basic serialization checks
    assert "git_commit" in meta
    assert "data_lineage_hash" in meta
    assert meta["python_version"] == sys.version
    assert isinstance(meta["package_versions"], dict)

    prov = model_store.get_provenance(version_id, store_dir=tmp_path)
    expected_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1])
        .decode()
        .strip()
    )
    data_path = Path(__file__).resolve().parents[1] / "data_versions.json"
    h = hashlib.sha256()
    h.update(data_path.read_bytes())
    expected_lineage = h.hexdigest()
    assert prov["git_commit"] == expected_commit
    assert prov["data_lineage_hash"] == expected_lineage
    assert prov["python_version"] == sys.version
    assert isinstance(prov["package_versions"], dict)
