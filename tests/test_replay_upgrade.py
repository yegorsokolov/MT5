import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_replay_updates_pnl_on_upgrade(tmp_path, monkeypatch):
    sys.modules.pop("pandas", None)
    import pandas as pd
    reports = tmp_path / "reports"
    reports.mkdir()
    trades = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "symbol": ["XYZ"],
            "pnl": [1.0],
        }
    )
    trades.to_csv(reports / "trades.csv", index=False)

    # metrics store path
    import analytics.metrics_store as ms

    monkeypatch.setattr(ms, "TS_PATH", tmp_path / "metrics.parquet", raising=False)
    monkeypatch.chdir(tmp_path)

    import types
    gbr = type("GBR", (), {})
    sklearn = types.SimpleNamespace(ensemble=types.SimpleNamespace(GradientBoostingRegressor=gbr))
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.ensemble", sklearn.ensemble)
    sys.modules.setdefault("joblib", types.SimpleNamespace(dump=lambda *a, **k: None, load=lambda *a, **k: {}))

    import analysis.replay as replay

    base = replay.reprocess(reports / "replays")
    base_pnl = base["pnl_new"].iloc[0]

    impact_df = pd.DataFrame(
        {
            "symbol": ["XYZ"],
            "timestamp": [pd.Timestamp("2024-01-01")],
            "impact": [0.5],
            "uncertainty": [0.1],
        }
    )
    impact_path = tmp_path / "impact.parquet"
    impact_df.to_parquet(impact_path, index=False)
    monkeypatch.setenv("NEWS_IMPACT_PATH", str(impact_path))
    import news.impact_model as im

    importlib.reload(im)

    out = reports / "replays"
    updated = replay.reprocess(out)
    assert updated["pnl_new"].iloc[0] > base_pnl
    assert (out / "old_vs_new_pnl.csv").exists()
    assert (out / "sharpe_deltas.csv").exists()
