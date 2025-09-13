import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.baseline import BaselineStrategy, RiskProfile


def test_profiles_yield_distinct_signals_and_thresholds():
    low = BaselineStrategy(
        atr_stop_long=100.0,
        trailing_stop_pct=1.0,
        risk_profile=RiskProfile(tolerance=0.5, leverage_cap=1.0, drawdown_limit=0.05),
    )
    high = BaselineStrategy(
        atr_stop_long=100.0,
        trailing_stop_pct=1.0,
        risk_profile=RiskProfile(tolerance=1.0, leverage_cap=1.0, drawdown_limit=0.2),
    )

    assert high._apply_risk(1) > low._apply_risk(1)

    low.latest_atr = high.latest_atr = 1.0
    low._open_long(100.0)
    high._open_long(100.0)

    exit_low = low._handle_open_position(94.0)
    exit_high = high._handle_open_position(94.0)

    assert exit_low == -1 and exit_high == 0
