import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
import pytest
from types import SimpleNamespace
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Configure modest streaming parameters so tests observe chunked behaviour
STREAM_CHUNK_SIZE = 25
STREAM_FEATURE_LOOKBACK = 5

# Avoid heavy environment checks during imports
sys.modules.setdefault(
    "utils.environment", types.SimpleNamespace(ensure_environment=lambda: None)
)

# Provide a lightweight fallback for ``data.history`` when optional
# dependencies are unavailable.
try:  # pragma: no cover - prefer real implementation when available
    history = __import__("data.history", fromlist=["load_history_iter"])
except Exception:  # pragma: no cover - fallback for minimal test environment
    history = types.ModuleType("data.history")

    def _stub_load_history_iter(path, chunk_size):
        frame = pd.read_pickle(path)
        for start in range(0, len(frame), chunk_size):
            yield frame.iloc[start : start + chunk_size]

    def _stub_load_history_parquet(path):
        return pd.read_pickle(path)

    def _stub_save_history_parquet(df, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path)

    history.load_history_iter = _stub_load_history_iter  # type: ignore[attr-defined]
    history.load_history_parquet = _stub_load_history_parquet  # type: ignore[attr-defined]
    history.save_history_parquet = _stub_save_history_parquet  # type: ignore[attr-defined]
    base_pkg = sys.modules.setdefault("data", types.ModuleType("data"))
    if not hasattr(base_pkg, "__path__"):
        base_pkg.__path__ = []  # type: ignore[attr-defined]
    setattr(base_pkg, "history", history)
    sys.modules["data.history"] = history
else:
    history = pytest.importorskip("data.history")

try:  # pragma: no cover - prefer real implementation
    streaming_mod = __import__("data.streaming", fromlist=["stream_features"])
except Exception:  # pragma: no cover - minimal fallback
    streaming_mod = types.ModuleType("data.streaming")

    def _stub_stream_features(frames, **kwargs):
        for frame in frames:
            yield frame

    def _stub_stream_labels(frames, horizons):
        labels_mod = sys.modules.get("data.labels")
        for frame in frames:
            if labels_mod is None or not hasattr(labels_mod, "multi_horizon_labels"):
                yield pd.DataFrame()
            else:
                yield labels_mod.multi_horizon_labels(frame["mid"], horizons)

    streaming_mod.stream_features = _stub_stream_features  # type: ignore[attr-defined]
    streaming_mod.stream_labels = _stub_stream_labels  # type: ignore[attr-defined]
    base_pkg = sys.modules.setdefault("data", types.ModuleType("data"))
    if not hasattr(base_pkg, "__path__"):
        base_pkg.__path__ = []  # type: ignore[attr-defined]
    setattr(base_pkg, "streaming", streaming_mod)
    sys.modules["data.streaming"] = streaming_mod
load_history_iter = history.load_history_iter
load_history_parquet = history.load_history_parquet

from training.data_loader import _collect_streaming_features

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def _make_history(path: Path, rows: int = 100) -> None:
    ts = pd.date_range("2020-01-01", periods=rows, freq="s")
    df = pd.DataFrame({
        "Timestamp": ts,
        "Bid": np.linspace(1.0, 1.0 + rows * 0.0001, rows),
        "Ask": np.linspace(1.0, 1.0 + rows * 0.0001, rows) + 0.0002,
    })
    df["spread"] = df["Ask"] - df["Bid"]
    history.save_history_parquet(df, path)


def _run_train(tmpdir: Path, stream: bool):
    train = pytest.importorskip("train")
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("joblib")

    class DummyClf(train.LGBMClassifier.__mro__[0]):
        def fit(self, X, y, **kwargs):  # type: ignore[override]
            kwargs.pop("early_stopping_rounds", None)
            kwargs.pop("eval_set", None)
            kwargs.pop("verbose", None)
            return super().fit(X, y)

    max_seen = 0

    def _simple_features(df: pd.DataFrame) -> pd.DataFrame:
        nonlocal max_seen
        df = df.copy()
        max_seen = max(max_seen, len(df))
        df["mid"] = (df["Bid"] + df["Ask"]) / 2
        df["return"] = df["Bid"].pct_change().fillna(0)
        df["spread"] = df["Ask"] - df["Bid"]
        for col in [
            "ma_5",
            "ma_10",
            "ma_30",
            "ma_60",
            "volatility_30",
            "rsi_14",
            "news_sentiment",
            "market_regime",
            "cross_mom_TEST_1",
            "cross_mom_TEST_2",
            "factor_1",
        ]:
            df[col] = 0.0
        return df

    train.make_features = _simple_features  # type: ignore
    train.LGBMClassifier = DummyClf  # type: ignore
    train.joblib.dump = lambda *a, **k: None  # type: ignore
    cfg = {
        "seed": 0,
        "risk_per_trade": 0.01,
        "symbols": ["TEST"],
        "n_splits": 2,
        "stream_history": stream,
        "stream_chunk_size": STREAM_CHUNK_SIZE,
        "stream_feature_lookback": STREAM_FEATURE_LOOKBACK,
        "use_scaler": False,
        "early_stopping_rounds": 5,
    }
    cfg_path = tmpdir / "config.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f)
    os.environ["CONFIG_FILE"] = str(cfg_path)
    mlflow.end_run()
    train.main()
    with open(ROOT / "classification_report.json") as f:
        return json.load(f), max_seen


def _run_train_nn(tmpdir: Path, stream: bool):
    train_nn = pytest.importorskip("train_nn")
    mlflow = pytest.importorskip("mlflow")
    joblib = pytest.importorskip("joblib")
    torch = pytest.importorskip("torch")

    def _simple_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["mid"] = (df["Bid"] + df["Ask"]) / 2
        df["return"] = df["Bid"].pct_change().fillna(0)
        df["spread"] = df["Ask"] - df["Bid"]
        for col in [
            "ma_5",
            "ma_10",
            "ma_30",
            "ma_60",
            "volatility_30",
            "rsi_14",
            "news_sentiment",
            "market_regime",
            "cross_mom_TEST_1",
            "cross_mom_TEST_2",
            "factor_1",
        ]:
            df[col] = 0.0
        return df

    train_nn.make_features = _simple_features  # type: ignore
    cfg = {
        "seed": 0,
        "risk_per_trade": 0.01,
        "symbols": ["TEST"],
        "stream_history": stream,
        "stream_chunk_size": STREAM_CHUNK_SIZE,
        "stream_feature_lookback": STREAM_FEATURE_LOOKBACK,
        "epochs": 1,
        "sequence_length": 5,
        "val_size": 0.2,
        "train_rows": 60,
    }
    cfg_path = tmpdir / "config.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f)
    os.environ["CONFIG_FILE"] = str(cfg_path)
    mlflow.end_run()
    train_nn.main()

    state = joblib.load(ROOT / "model_transformer.pt")
    feat_path = ROOT / "selected_features.json"
    features = json.loads(feat_path.read_text())
    df = _simple_features(load_history_parquet(DATA_DIR / "TEST_history.parquet"))
    df["Symbol"] = "TEST"
    df["SymbolCode"] = 0
    train_rows = cfg.get("train_rows", 40)
    total_rows = len(df)
    desired_val = max(1, total_rows - train_rows)
    folds = train_nn.generate_time_series_folds(
        total_rows,
        n_splits=cfg.get("cv_splits", 1),
        test_size=desired_val,
        embargo=cfg.get("cv_embargo", 0),
        min_train_size=cfg.get("cv_min_train_size"),
        group_gap=cfg.get("cv_group_gap", 0),
        groups=train_nn.resolve_group_labels(df),
    )
    train_idx, test_idx = folds[-1]
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    X_test, _ = train_nn.make_sequence_arrays(
        test_df, features, cfg.get("sequence_length", 5)
    )
    model = train_nn.TransformerModel(len(features), num_symbols=1, num_regimes=1)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    return preds


def test_collect_streaming_features_is_lazy(monkeypatch, tmp_path):
    chunks = [
        pd.DataFrame({"mid": [0.1, 0.2], "Symbol": "LAZY"}),
        pd.DataFrame({"mid": [0.3], "Symbol": "LAZY"}),
    ]
    calls = {"feature": 0}

    def fake_symbol_history_chunks(symbol, cfg, root, *, chunk_size, validate):
        assert symbol == "LAZY"
        assert chunk_size == 4
        return iter(chunks)

    monkeypatch.setattr(
        "training.data_loader._symbol_history_chunks", fake_symbol_history_chunks
    )

    def fake_stream_features(frames, **kwargs):
        for frame in frames:
            calls["feature"] += 1
            yield frame.assign(streamed=True)

    monkeypatch.setattr("data.streaming.stream_features", fake_stream_features)

    cfg = SimpleNamespace(strategy=SimpleNamespace(symbols=["LAZY"]))
    iterator = _collect_streaming_features(
        cfg.strategy.symbols,
        cfg,
        tmp_path,
        chunk_size=4,
        feature_lookback=2,
        validate=False,
    )

    assert calls["feature"] == 0
    first = next(iterator)
    assert calls["feature"] == 1
    assert first["streamed"].all()

    remaining = list(iterator)
    assert calls["feature"] == len(chunks)
    assert len(remaining) == len(chunks) - 1
    total_rows = len(first) + sum(len(chunk) for chunk in remaining)
    assert total_rows == sum(len(chunk) for chunk in chunks)


def test_load_history_iter_equivalent(tmp_path):
    path = tmp_path / "hist.parquet"
    _make_history(path, rows=50)
    full = load_history_parquet(path)
    streamed = pd.concat(list(load_history_iter(path, 20)), ignore_index=True)
    pd.testing.assert_frame_equal(full, streamed)


def test_train_stream_equivalence(tmp_path):
    hist_path = DATA_DIR / "TEST_history.parquet"
    _make_history(hist_path, rows=80)
    res_full, max_full = _run_train(tmp_path, stream=False)
    res_stream, max_stream = _run_train(tmp_path, stream=True)
    assert res_full == res_stream
    assert max_stream <= STREAM_CHUNK_SIZE + STREAM_FEATURE_LOOKBACK
    assert max_stream < max_full


def test_train_nn_stream_equivalence(tmp_path):
    hist_path = DATA_DIR / "TEST_history.parquet"
    _make_history(hist_path, rows=80)
    model_full = _run_train_nn(tmp_path, stream=False)
    model_stream = _run_train_nn(tmp_path, stream=True)
    np.testing.assert_allclose(model_full, model_stream)
