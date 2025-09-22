from __future__ import annotations

import pandas as pd

from mt5 import monitor_drift


def _make_batch(start: int) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame({"f": [start, start + 1]})
    preds = pd.Series([start / 10, (start + 1) / 10])
    return features, preds


def test_record_appends_without_pyarrow(tmp_path, monkeypatch):
    store_path = tmp_path / "drift.parquet"
    monitor = monitor_drift.DriftMonitor(store_path=store_path)
    monkeypatch.setattr(monitor_drift, "_pyarrow_engine", lambda: None)

    features1, preds1 = _make_batch(0)
    features2, preds2 = _make_batch(2)

    monitor.record(features1, preds1)
    monitor.record(features2, preds2)

    persisted = pd.read_parquet(store_path)
    assert len(persisted) == 4
    assert list(persisted["prediction"]) == [0.0, 0.1, 0.2, 0.3]


def test_record_appends_with_pyarrow(monkeypatch, tmp_path):
    store_path = tmp_path / "drift.parquet"
    monitor = monitor_drift.DriftMonitor(store_path=store_path)

    monkeypatch.setattr(monitor_drift, "_pyarrow_engine", lambda: "pyarrow")

    calls: list[tuple[str, object]] = []

    orig_to_parquet = pd.DataFrame.to_parquet
    def fake_to_parquet(self, path, *args, **kwargs):  # type: ignore[no-untyped-def]
        engine = kwargs.get("engine")
        calls.append(("write", engine))
        try:
            return orig_to_parquet(self, path, *args, **kwargs)
        except ImportError:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("engine", None)
            return orig_to_parquet(self, path, *args, **fallback_kwargs)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    orig_read_parquet = pd.read_parquet
    def fake_read_parquet(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        engine = kwargs.get("engine")
        calls.append(("read", engine))
        try:
            return orig_read_parquet(path, *args, **kwargs)
        except ImportError:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("engine", None)
            return orig_read_parquet(path, *args, **fallback_kwargs)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    features1, preds1 = _make_batch(0)
    features2, preds2 = _make_batch(2)

    monitor.record(features1, preds1)
    monitor.record(features2, preds2)

    persisted = orig_read_parquet(store_path)
    assert len(persisted) == 4
    assert list(persisted["prediction"]) == [0.0, 0.1, 0.2, 0.3]

    write_calls = [engine for kind, engine in calls if kind == "write"]
    read_calls = [engine for kind, engine in calls if kind == "read"]
    assert write_calls.count("pyarrow") == 2
    assert "pyarrow" in read_calls
