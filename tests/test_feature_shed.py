import types
import sys
import pandas as pd
import numpy as np

from pathlib import Path
import sys
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("requests", types.SimpleNamespace(get=lambda *a, **k: None))
sklearn_stub = types.ModuleType("sklearn")
sk_decomp = types.ModuleType("sklearn.decomposition")
class _PCA:
    def __init__(self, *a, **k):
        pass
sk_decomp.PCA = _PCA
sklearn_stub.decomposition = sk_decomp
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.decomposition", sk_decomp)
pywt_stub = types.SimpleNamespace(wavedec=lambda data, wavelet, level: [np.zeros_like(data)] * (level + 1))
sys.modules.setdefault("pywt", pywt_stub)
sys.modules.setdefault("duckdb", types.SimpleNamespace(connect=lambda *a, **k: None))
sys.modules.setdefault("networkx", types.SimpleNamespace())

import data.features as feat
sys.modules.setdefault("regime", types.SimpleNamespace(label_regimes=lambda df: df))


def test_degradable_features_shed(monkeypatch):
    metrics = []
    monkeypatch.setattr(feat, "record_metric", lambda name, value, tags=None: metrics.append((name, tags)))
    monitor_stub = types.SimpleNamespace(
        capability_tier="standard",
        tick_to_signal_latency=1.0,
        capabilities=types.SimpleNamespace(cpus=8),
    )
    monkeypatch.setattr(feat, "monitor", monitor_stub)

    utils_stub = types.SimpleNamespace(
        load_config=lambda: {
            "use_dtw": False,
            "use_kalman": False,
            "use_atr": False,
            "use_donchian": False,
            "latency_threshold": 0.1,
        }
    )
    sys.modules["utils"] = utils_stub

    monkeypatch.setattr(feat, "add_session_features", lambda df: df)
    monkeypatch.setattr(feat, "add_economic_calendar_features", lambda df: df)
    monkeypatch.setattr(feat, "add_news_sentiment_features", lambda df: df)
    monkeypatch.setattr(feat, "add_index_features", lambda df: df)
    monkeypatch.setattr(feat, "add_cross_asset_features", lambda df: df)
    monkeypatch.setattr(feat, "load_macro_series", lambda symbols: pd.DataFrame())
    monkeypatch.setattr(feat, "optimize_dtypes", lambda df: df)

    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024", periods=5, freq="T"),
            "Bid": np.arange(5) + 1.0,
            "Ask": np.arange(5) + 1.1,
        }
    )

    out = feat.make_features(df)
    assert "garch_vol" in out.columns
    assert out["garch_vol"].isna().all()
    assert any(m[0] == "feature_shed" and m[1]["feature"] == "garch_volatility" for m in metrics)
