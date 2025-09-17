import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Configure modest streaming parameters so tests observe chunked behaviour
STREAM_CHUNK_SIZE = 25
STREAM_FEATURE_LOOKBACK = 5

# Avoid heavy environment checks during imports
import types
sys.modules.setdefault(
    "utils.environment", types.SimpleNamespace(ensure_environment=lambda: None)
)

history = pytest.importorskip("data.history")
load_history_iter = history.load_history_iter
load_history_parquet = history.load_history_parquet

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
    df.to_parquet(path, index=False)


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
