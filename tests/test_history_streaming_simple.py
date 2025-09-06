from pathlib import Path
import pandas as pd
import numpy as np
import types, sys, importlib.util

# Stub minimal dependencies so data.history can be imported without optional packages
utils = types.ModuleType('utils')
utils.data_backend = types.SimpleNamespace(get_dataframe_module=lambda: pd)
utils.resource_monitor = types.SimpleNamespace(monitor=types.SimpleNamespace(latest_usage={}))
analysis = types.ModuleType('analysis')
analysis.data_quality = types.SimpleNamespace(apply_quality_checks=lambda df: (df, {}))
analytics = types.ModuleType('analytics')
analytics.metrics_store = types.SimpleNamespace(record_metric=lambda *a, **k: None)

data_pkg = types.ModuleType('data')
data_pkg.__path__ = []
data_delta_store = types.SimpleNamespace(DeltaStore=lambda: types.SimpleNamespace(ingest=lambda path, df: pd.DataFrame()))
data_expectations = types.SimpleNamespace(validate_dataframe=lambda df, name: None)
data_versioning = types.SimpleNamespace(compute_hash=lambda path: 'hash')

sys.modules.update({
    'utils': utils,
    'utils.data_backend': utils.data_backend,
    'utils.resource_monitor': utils.resource_monitor,
    'analysis': analysis,
    'analysis.data_quality': analysis.data_quality,
    'analytics': analytics,
    'analytics.metrics_store': analytics.metrics_store,
    'data': data_pkg,
    'data.delta_store': data_delta_store,
    'data.expectations': data_expectations,
    'data.versioning': data_versioning,
})

spec = importlib.util.spec_from_file_location('data.history', Path(__file__).resolve().parents[1] / 'data' / 'history.py')
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)
load_history_parquet = history.load_history_parquet
load_history_iter = history.load_history_iter


def _make_history(path: Path, rows: int = 100) -> None:
    ts = pd.date_range('2020-01-01', periods=rows, freq='s')
    df = pd.DataFrame({'Timestamp': ts,
                       'Bid': np.linspace(1.0, 1.0 + rows * 0.0001, rows),
                       'Ask': np.linspace(1.0, 1.0 + rows * 0.0001, rows) + 0.0002})
    df.to_parquet(path, index=False)


def test_iter_matches_full(tmp_path: Path) -> None:
    path = tmp_path / 'hist.parquet'
    _make_history(path, rows=50)
    full = load_history_parquet(path)
    streamed = pd.concat(list(load_history_iter(path, 20)), ignore_index=True)
    pd.testing.assert_frame_equal(full, streamed)
