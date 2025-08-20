import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


def setup_impact(tmp_path, impact):
    path = tmp_path / "impact.parquet"
    df = pd.DataFrame({
        "symbol": ["XYZ"],
        "timestamp": [pd.Timestamp("2024-01-01")],
        "impact": [impact],
        "uncertainty": [0.1],
    })
    df.to_parquet(path, index=False)
    return path


def test_news_impact_reduces_drawdown(tmp_path, monkeypatch):
    path = setup_impact(tmp_path, -0.9)
    monkeypatch.setenv("NEWS_IMPACT_PATH", str(path))
    importlib.reload(importlib.import_module("news.impact_model"))
    rm_mod = importlib.reload(importlib.import_module("risk_manager"))
    rm = rm_mod.RiskManager(max_drawdown=100)
    size = rm.adjust_size("XYZ", 100.0, pd.Timestamp("2024-01-01"), direction=1)
    if size > 0:
        rm.update("bot", -50)
    loss_filtered = rm.metrics.daily_loss
    rm_no = rm_mod.RiskManager(max_drawdown=100)
    rm_no.update("bot", -50)
    assert loss_filtered > rm_no.metrics.daily_loss


def test_news_impact_boosts_size(tmp_path, monkeypatch):
    path = setup_impact(tmp_path, 0.9)
    monkeypatch.setenv("NEWS_IMPACT_PATH", str(path))
    importlib.reload(importlib.import_module("news.impact_model"))
    rm_mod = importlib.reload(importlib.import_module("risk_manager"))
    rm_mod._IMPACT_THRESHOLD = 0.5
    rm_mod._IMPACT_BOOST = 2.0
    rm = rm_mod.RiskManager(max_drawdown=100)
    size = rm.adjust_size("XYZ", 10.0, pd.Timestamp("2024-01-01"), direction=1)
    assert size == pytest.approx(20.0)
