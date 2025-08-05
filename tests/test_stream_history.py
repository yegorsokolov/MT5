import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Avoid heavy environment checks during imports
import types
sys.modules.setdefault("utils.environment", types.SimpleNamespace(ensure_environment=lambda: None))
import mlflow
import joblib
import torch

from data.history import load_history_iter, load_history_parquet
import train
import train_nn

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
    class DummyClf(train.LGBMClassifier.__mro__[0]):
        def fit(self, X, y, **kwargs):  # type: ignore[override]
            kwargs.pop("early_stopping_rounds", None)
            kwargs.pop("eval_set", None)
            kwargs.pop("verbose", None)
            return super().fit(X, y)

    def _simple_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
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
        "stream_chunk_size": 1000,
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
        return json.load(f)


def _run_train_nn(tmpdir: Path, stream: bool):
    def _simple_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
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
        "stream_chunk_size": 1000,
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
    features = [
        "return",
        "ma_5",
        "ma_10",
        "ma_30",
        "ma_60",
        "volatility_30",
        "spread",
        "rsi_14",
        "news_sentiment",
        "market_regime",
        "cross_mom_TEST_1",
        "cross_mom_TEST_2",
        "factor_1",
        "SymbolCode",
    ]
    df = _simple_features(load_history_parquet(DATA_DIR / "TEST_history.parquet"))
    df["Symbol"] = "TEST"
    df["SymbolCode"] = 0
    train_df, test_df = train_nn.train_test_split(df, cfg.get("train_rows", 40))
    X_test, _ = train_nn.make_sequence_arrays(test_df, features, cfg.get("sequence_length", 5))
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
    res_full = _run_train(tmp_path, stream=False)
    res_stream = _run_train(tmp_path, stream=True)
    assert res_full == res_stream


def test_train_nn_stream_equivalence(tmp_path):
    hist_path = DATA_DIR / "TEST_history.parquet"
    _make_history(hist_path, rows=80)
    model_full = _run_train_nn(tmp_path, stream=False)
    model_stream = _run_train_nn(tmp_path, stream=True)
    np.testing.assert_allclose(model_full, model_stream)
