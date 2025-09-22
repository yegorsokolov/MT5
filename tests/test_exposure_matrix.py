import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mt5.risk_manager import RiskManager


def test_exposure_matrix_updates_and_limits(tmp_path):
    rm = RiskManager(max_drawdown=1000, max_long_exposure=100, max_short_exposure=100)
    # open positions in two assets
    rm.update("bot", pnl=0.0, exposure=50.0, symbol="AAA")
    rm.update("bot", pnl=0.0, exposure=50.0, symbol="BBB")
    net = {
        s: rm.net_exposure.long.get(s, 0.0) - rm.net_exposure.short.get(s, 0.0)
        for s in set(rm.net_exposure.long) | set(rm.net_exposure.short)
    }
    mat_low = rm.exposure_matrix.weighted_exposures(net)
    allowed_low = rm.net_exposure.limit("AAA", 1000)
    assert mat_low.loc["AAA", "BBB"] == 0.0
    assert allowed_low > 0

    # feed highly correlated returns
    for r in [0.01, 0.02, 0.03, 0.04, 0.05]:
        rm.record_returns({"AAA": r, "BBB": r})
    net = {
        s: rm.net_exposure.long.get(s, 0.0) - rm.net_exposure.short.get(s, 0.0)
        for s in set(rm.net_exposure.long) | set(rm.net_exposure.short)
    }
    mat_high = rm.exposure_matrix.weighted_exposures(net)
    allowed_high = rm.net_exposure.limit("AAA", 1000)
    assert mat_high.loc["AAA", "BBB"] == pytest.approx(2500, rel=1e-2)
    assert allowed_high < allowed_low
    assert allowed_high < 1.0
