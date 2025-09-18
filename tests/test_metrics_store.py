import importlib
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

# ``tests.conftest`` pre-populates a stub for ``analytics.metrics_store``.
# Replace it with the actual implementation for these integration tests.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.pop("analytics.metrics_store", None)
metrics_store = importlib.import_module("analytics.metrics_store")

MetricsStore = metrics_store.MetricsStore
record_metric = metrics_store.record_metric
query_metrics = metrics_store.query_metrics


def _assert_near_constant(durations: list[float], *, warmup: int = 5, factor: float = 4.0) -> None:
    assert durations, "expected durations to be collected"
    effective = durations[warmup:] if len(durations) > warmup else durations
    slice_len = max(5, len(effective) // 10)
    first = statistics.median(effective[:slice_len])
    last = statistics.median(effective[-slice_len:])
    assert last <= first * factor, f"append time regressed: first={first}, last={last}"


def test_metrics_store_append_constant_time(tmp_path: Path) -> None:
    path = tmp_path / "metrics.parquet"
    store = MetricsStore(path=path)

    start = pd.Timestamp("2024-01-01")
    durations: list[float] = []

    for offset in range(200):
        date = start + pd.Timedelta(days=offset)
        begin = time.perf_counter()
        store.append(
            date,
            ret=float(offset) / 10.0,
            sharpe=1.5 + offset / 100.0,
            drawdown=-0.1,
            regime=offset % 3,
        )
        durations.append(time.perf_counter() - begin)

    frame = store.load()
    assert len(frame) == 200
    assert frame.index.is_monotonic_increasing
    assert set(frame.columns) == {"return", "sharpe", "drawdown", "regime"}

    # Upsert behaviour keeps the latest value for a day.
    store.append(start, ret=999.0, sharpe=9.9, drawdown=-0.5, regime="override")
    frame = store.load()
    assert frame.loc[start, "return"] == pytest.approx(999.0)
    assert frame.loc[start, "regime"] == "override"

    _assert_near_constant(durations)


def test_record_metric_incremental_appends(tmp_path: Path) -> None:
    path = tmp_path / "metrics_timeseries.parquet"
    durations: list[float] = []

    for step in range(320):
        tags = {"fold": step % 5}
        if step % 40 == 0:
            tags["model"] = f"model-{step % 7}"
        begin = time.perf_counter()
        record_metric("loss", float(step) / 100.0, tags=tags, path=path)
        durations.append(time.perf_counter() - begin)

    # Introduce a new column on the fly.
    record_metric("loss", 0.0, tags={"fold": 0, "phase": "final"}, path=path)

    frame = query_metrics(path=path)
    assert len(frame) == 321
    assert frame["name"].unique().tolist() == ["loss"]
    assert "fold" in frame.columns
    assert frame["fold"].notna().all()
    assert "model" in frame.columns
    assert frame["model"].notna().sum() == 8  # 320 / 40
    assert "phase" in frame.columns
    assert frame["phase"].dropna().tolist() == ["final"]

    # Filtering on tags still works with the dataset-backed store.
    filtered = query_metrics(name="loss", tags={"fold": 2}, path=path)
    assert not filtered.empty
    assert set(filtered["fold"].dropna().unique()) == {2}

    _assert_near_constant(durations)
