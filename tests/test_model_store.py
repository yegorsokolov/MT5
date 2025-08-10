import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "model_store", Path(__file__).resolve().parents[1] / "models" / "model_store.py"
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
