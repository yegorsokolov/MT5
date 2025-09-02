import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.resource_monitor import monitor
import importlib.util
import types
import sys

# Stub out ``features`` and ``analysis`` packages and create a minimal
# ``data`` package so ``data.features`` can be imported without executing the
# real heavy initialisers.
features_stub = types.ModuleType("features")
features_stub.get_feature_pipeline = lambda: [lambda df: df]
news_stub = types.ModuleType("features.news")
news_stub.add_economic_calendar_features = lambda df: df
news_stub.add_news_sentiment_features = lambda df: df
cross_stub = types.ModuleType("features.cross_asset")
cross_stub.add_index_features = lambda df: df
cross_stub.add_cross_asset_features = lambda df: df
analysis_stub = types.ModuleType("analysis.feature_gate")
analysis_stub.select = lambda df, tier, regime, persist=True: (df, [])

data_pkg = types.ModuleType("data")
sys.modules.update(
    {
        "features": features_stub,
        "features.news": news_stub,
        "features.cross_asset": cross_stub,
        "analysis": types.ModuleType("analysis"),
        "analysis.feature_gate": analysis_stub,
        "data": data_pkg,
    }
)
sys.modules["analysis"].feature_gate = analysis_stub

# Load ``data.corporate_actions`` and ``data.features`` modules manually
spec_ca = importlib.util.spec_from_file_location(
    "data.corporate_actions", ROOT / "data" / "corporate_actions.py"
)
corp_actions = importlib.util.module_from_spec(spec_ca)
spec_ca.loader.exec_module(corp_actions)
sys.modules["data.corporate_actions"] = corp_actions

spec = importlib.util.spec_from_file_location(
    "data.features", ROOT / "data" / "features.py"
)
feature_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_module)
sys.modules["data.features"] = feature_module
make_features = feature_module.make_features


def _write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)


def test_corporate_actions_alignment(tmp_path, monkeypatch):
    _write_csv(
        tmp_path / "dataset" / "corporate_actions" / "dividends" / "AAA.csv",
        ["Date", "dividend"],
        [["2020-01-01", 0.5]],
    )
    _write_csv(
        tmp_path / "dataset" / "corporate_actions" / "splits" / "AAA.csv",
        ["Date", "split"],
        [["2020-01-01", 2.0]],
    )
    _write_csv(
        tmp_path / "dataset" / "corporate_actions" / "insider" / "AAA.csv",
        ["Date", "insider_trades"],
        [["2020-01-01", 100]],
    )
    _write_csv(
        tmp_path / "dataset" / "corporate_actions" / "thirteenf" / "AAA.csv",
        ["Date", "institutional_holdings"],
        [["2020-01-01", 200]],
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(feature_module, "get_feature_pipeline", lambda: [lambda df: df])
    monitor.capability_tier = "hpc"

    base = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2020-01-02", "2020-01-03"], utc=True),
            "Symbol": ["AAA", "AAA"],
            "return": [0.0, 0.0],
            "ma_5": [0.0, 0.0],
            "ma_10": [0.0, 0.0],
            "ma_30": [0.0, 0.0],
            "ma_60": [0.0, 0.0],
            "volatility_30": [0.0, 0.0],
            "rsi_14": [0.0, 0.0],
            "market_regime": [0, 0],
        }
    )

    out = make_features(base, validate=True)
    assert (out["dividend"] == 0.5).all()
    assert (out["split"] == 2.0).all()
    assert (out["insider_trades"] == 100).all()
    assert (out["institutional_holdings"] == 200).all()


def test_corporate_actions_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(feature_module, "get_feature_pipeline", lambda: [lambda df: df])
    monitor.capability_tier = "hpc"

    base = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2020-01-02"], utc=True),
            "Symbol": ["BBB"],
            "return": [0.0],
            "ma_5": [0.0],
            "ma_10": [0.0],
            "ma_30": [0.0],
            "ma_60": [0.0],
            "volatility_30": [0.0],
            "rsi_14": [0.0],
            "market_regime": [0],
        }
    )

    out = make_features(base, validate=True)
    for col in ["dividend", "split", "insider_trades", "institutional_holdings"]:
        assert col in out.columns
        assert (out[col] == 0.0).all()
