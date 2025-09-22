import sys
from pathlib import Path
import pandas as pd
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("mlflow", types.SimpleNamespace(
    set_tracking_uri=lambda *a, **k: None,
    set_experiment=lambda *a, **k: None,
    start_run=lambda *a, **k: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, *exc: None),
    log_dict=lambda *a, **k: None,
))
sys.modules.setdefault("utils.environment", types.SimpleNamespace(ensure_environment=lambda: None))
sklearn_stub = types.ModuleType("sklearn")
sklearn_stub.decomposition = types.SimpleNamespace(PCA=lambda *a, **k: None)
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.decomposition", sklearn_stub.decomposition)
sys.modules.setdefault("duckdb", types.SimpleNamespace(connect=lambda *a, **k: None))
sys.modules.setdefault("networkx", types.SimpleNamespace())

from data.history import save_history_parquet, load_history_iter
from mt5.scheduler import vacuum_history


def _make_df(start, periods):
    ts = pd.date_range(start, periods=periods, freq="min")
    return pd.DataFrame({"Timestamp": ts, "Bid": range(periods), "Ask": range(periods)})


def test_partitioned_lazy_read(tmp_path):
    path = tmp_path / "hist.parquet"
    df1 = _make_df("2020-01-01", 2)
    df2 = _make_df("2020-01-02", 2)
    save_history_parquet(df1, path)
    save_history_parquet(df2, path)
    chunks = list(load_history_iter(path, 1))
    assert len(chunks) == 4
    loaded = pd.concat(chunks, ignore_index=True)
    expected = pd.concat([df1, df2], ignore_index=True)
    pd.testing.assert_frame_equal(loaded.sort_values("Timestamp").reset_index(drop=True), expected)


def test_vacuum_removes_old_files(tmp_path):
    path = tmp_path / "hist.parquet"
    df1 = _make_df("2020-01-01", 2)
    df2 = _make_df("2020-01-01", 2)
    save_history_parquet(df1, path)
    # Manually add another small file to the same partition to simulate fragmentation
    part_dir = next(path.glob("date=2020-01-01"))
    import pyarrow as pa, pyarrow.parquet as pq
    extra = pa.table({"Timestamp": [pd.Timestamp("2020-01-01")], "Bid": [0], "Ask": [0], "date": ["2020-01-01"]})
    pq.write_table(extra, part_dir / "extra.parquet", compression="zstd")
    save_history_parquet(df2, path)
    assert len(list(part_dir.glob("*.parquet"))) > 1
    vacuum_history(path)
    files = list(part_dir.glob("*.parquet"))
    assert len(files) == 1
