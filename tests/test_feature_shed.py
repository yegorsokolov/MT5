import types
import sys
import pandas as pd
import numpy as np

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
stub = types.ModuleType("data")
stub.__path__ = [str(DATA_ROOT)]
sys.modules.setdefault("data", stub)
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

utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda: {}
rm_stub = types.ModuleType("utils.resource_monitor")
rm_stub.ResourceCapabilities = types.SimpleNamespace
caps = types.SimpleNamespace(cpus=8, memory_gb=16, has_gpu=False, gpu_count=0)
rm_stub.monitor = types.SimpleNamespace(
    capability_tier="standard", capabilities=caps, subscribe=lambda: types.SimpleNamespace()
)
sys.modules.setdefault("utils", utils_stub)
sys.modules.setdefault("utils.resource_monitor", rm_stub)

analysis_stub = types.ModuleType("analysis")
analysis_stub.__path__ = [str(Path(__file__).resolve().parents[1] / "analysis")]
fg_stub = types.ModuleType("analysis.feature_gate")
fg_stub.select = lambda df, tier, regime_id, persist=False: (df, [])
sys.modules.setdefault("analysis", analysis_stub)
sys.modules.setdefault("analysis.feature_gate", fg_stub)

import data.features as feat
sys.modules.setdefault("regime", types.SimpleNamespace(label_regimes=lambda df: df))


def test_degradable_features_shed(monkeypatch):
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

    monkeypatch.setattr(feat, "add_economic_calendar_features", lambda df: df)
    monkeypatch.setattr(feat, "add_news_sentiment_features", lambda df: df)
    monkeypatch.setattr(feat, "add_index_features", lambda df: df)
    monkeypatch.setattr(feat, "add_cross_asset_features", lambda df: df)

    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024", periods=5, freq="T"),
            "Bid": np.arange(5) + 1.0,
            "Ask": np.arange(5) + 1.1,
        }
    )

    out = feat.make_features(df)
    assert "mid" in out.columns
