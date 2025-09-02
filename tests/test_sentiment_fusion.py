import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sys.path.append(str(Path(__file__).resolve().parents[1]))

import types

sys.modules.setdefault("utils.environment", types.SimpleNamespace(ensure_environment=lambda: None))
sys.modules.setdefault("duckdb", types.SimpleNamespace())
sys.modules.setdefault("mlflow", types.SimpleNamespace())
sys.modules.setdefault("pandera", types.SimpleNamespace())
sys.modules.setdefault("gdown", types.SimpleNamespace())
sys.modules.setdefault("GitPython", types.SimpleNamespace())
sys.modules.setdefault("psutil", types.SimpleNamespace())
sys.modules.setdefault("lightgbm", types.SimpleNamespace())
sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 0)),
)
sys.modules.setdefault("tqdm", types.SimpleNamespace())
sys.modules.setdefault("river", types.SimpleNamespace())
sys.modules.setdefault("kafka", types.SimpleNamespace())
sys.modules.setdefault("redis", types.SimpleNamespace())
sys.modules.setdefault("uvloop", types.SimpleNamespace())
sys.modules.setdefault("shap", types.SimpleNamespace())
sys.modules.setdefault("statsmodels", types.SimpleNamespace())
sys.modules.setdefault("ydata_synthetic", types.SimpleNamespace())
sys.modules.setdefault("tsfresh", types.SimpleNamespace())
sys.modules.setdefault("exchange_calendars", types.SimpleNamespace())
sys.modules.setdefault("arch", types.SimpleNamespace())
sys.modules.setdefault("grpc", types.SimpleNamespace())
sys.modules.setdefault("grpc_tools", types.SimpleNamespace())
sys.modules.setdefault("backtrader", types.SimpleNamespace())
sys.modules.setdefault("freqtrade", types.SimpleNamespace())
sys.modules.setdefault("opentelemetry", types.SimpleNamespace())
sys.modules.setdefault("opentelemetry.sdk", types.SimpleNamespace())
sys.modules.setdefault("opentelemetry.exporter", types.SimpleNamespace())
sys.modules.setdefault("opentelemetry.instrumentation.logging", types.SimpleNamespace())
sys.modules.setdefault("networkx", types.SimpleNamespace())

from news import sentiment_fusion
from data.features import add_news_sentiment_features
from utils.resource_monitor import monitor


def test_sentiment_fusion_improves_metrics(tmp_path, monkeypatch):
    # patch paths for persistence
    monkeypatch.setattr(sentiment_fusion, "_DATA_PATH", tmp_path / "fused.parquet")
    monkeypatch.setattr(sentiment_fusion, "_MODEL_PATH", tmp_path / "model.pkl")

    rng = np.random.RandomState(0)
    n = 100
    embeddings = rng.randn(n, 5)
    surprise = rng.randn(n)
    target = 0.6 * embeddings.mean(axis=1) + 0.4 * surprise + rng.randn(n) * 0.1

    events = pd.DataFrame(
        {
            "symbol": ["EURUSD"] * n,
            "timestamp": pd.date_range("2021-01-01", periods=n, freq="H"),
            "embedding": list(embeddings),
            "surprise": surprise,
        }
    )

    # Baseline using only surprise
    base = RandomForestRegressor(n_estimators=50, random_state=0)
    base.fit(surprise.reshape(-1, 1), target)
    base_pred = base.predict(surprise.reshape(-1, 1))
    base_mse = mean_squared_error(target, base_pred)

    # Fused model
    sentiment_fusion.train(events, target)
    fused_pred = sentiment_fusion.score(events)["fused_sentiment"]
    fused_mse = mean_squared_error(target, fused_pred)

    assert fused_mse < base_mse

    # Integration with feature pipeline
    monkeypatch.setattr(monitor, "capability_tier", "standard")
    csv_dir = Path(__file__).resolve().parents[1] / "data" / "data"
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "news_sentiment.csv").write_text("Timestamp,sentiment\n2021-01-01,0\n")
    df = pd.DataFrame(
        {
            "Timestamp": events["timestamp"],
            "Symbol": events["symbol"],
            "mid": rng.randn(n),
        }
    )
    enriched = add_news_sentiment_features(df)
    assert "event_sentiment" in enriched.columns
    assert enriched["event_sentiment"].notna().any()
