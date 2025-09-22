import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mt5.risk_manager import RiskManager
from risk.tail_hedger import TailHedger


def test_risk_manager_tail_integration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rm = RiskManager(max_drawdown=0.4, initial_capital=1.0, tail_prob_limit=0.1)
    hedger = TailHedger(rm, var_threshold=1e9)
    rm.attach_tail_hedger(hedger)

    rng = np.random.default_rng(0)
    losses = -(rng.pareto(2, size=500) * 0.4)
    for l in losses:
        rm.update("bot", float(l))

    assert rm.metrics.tail_prob > rm.tail_prob_limit
    scaled = rm.adjust_position_size("bot", 1.0)
    assert scaled < 1.0
    assert rm.tail_hedger.hedge_ratio > 1.0
    log_file = Path("reports/tail_risk/evt_log.csv")
    assert log_file.exists()
    df = pd.read_csv(log_file)
    assert df["breach"].any()
