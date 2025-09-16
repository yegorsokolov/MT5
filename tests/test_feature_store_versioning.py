from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

import feature_store


class DummySeries:
    def __init__(self, values):
        self.values = list(values)
        self.dtype = type(self.values[0]).__name__ if self.values else "object"


class DummyFrame:
    def __init__(self, data):
        self._data = {k: list(v) for k, v in data.items()}
        self.columns = list(self._data.keys())
        self._series = {k: DummySeries(v) for k, v in self._data.items()}

    def __len__(self):
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def __getitem__(self, key):
        return self._series[key]

    def to_pickle(self, path: Path) -> None:
        with path.open("wb") as fh:
            pickle.dump(self._data, fh)

    def as_dict(self):
        return {k: list(v) for k, v in self._data.items()}


def test_feature_store_persists_metadata_and_code_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(feature_store, "INDEX_FILE", tmp_path / "index.json")
    def read_pickle(path: Path) -> DummyFrame:
        with path.open("rb") as fh:
            return DummyFrame(pickle.load(fh))

    monkeypatch.setattr(
        feature_store,
        "pd",
        SimpleNamespace(read_pickle=read_pickle),
    )

    df = DummyFrame({"value": [1, 2, 3]})

    feature_store.register_feature("v1", df, {"note": "first"}, code_hash="hash1")
    loaded_df, metadata = feature_store.load_feature("v1", with_metadata=True)
    assert loaded_df.as_dict() == df.as_dict()
    assert metadata["rows"] == len(df)
    assert metadata["columns"] == ["value"]
    assert metadata["code_hash"] == "hash1"
    assert metadata["note"] == "first"

    feature_store.register_feature("v2", df, {"note": "second", "code_hash": "hash2"})
    assert feature_store.latest_version() == "v2"

    versions = feature_store.list_versions()
    assert versions["v1"]["code_hash"] == "hash1"
    assert versions["v2"]["code_hash"] == "hash2"
