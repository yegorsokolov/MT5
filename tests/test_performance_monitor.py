import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategy.performance_monitor import PerformanceMonitor


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_disable_and_auto_reenable():
    clock = Clock()
    alerts: list[str] = []
    monitor = PerformanceMonitor(
        window=3,
        sharpe_threshold=0.0,
        win_rate_threshold=0.5,
        cooldown=10,
        clock=clock,
        alert_func=alerts.append,
    )

    for _ in range(3):
        monitor.record_trade("algo", -1.0)
    assert not monitor.is_enabled("algo")
    assert any("disabled" in msg for msg in alerts)

    for _ in range(3):
        monitor.record_trade("algo", 1.0)
    assert not monitor.is_enabled("algo")  # still in cooldown

    clock.advance(11)
    monitor.record_trade("algo", 1.0)
    assert monitor.is_enabled("algo")
    assert any("re-enabled" in msg for msg in alerts)


def test_manual_override():
    clock = Clock()
    alerts: list[str] = []
    monitor = PerformanceMonitor(
        window=3,
        sharpe_threshold=0.0,
        win_rate_threshold=0.5,
        cooldown=10,
        clock=clock,
        alert_func=alerts.append,
    )

    for _ in range(3):
        monitor.record_trade("algo", -1.0)
    assert not monitor.is_enabled("algo")

    monitor.override_enable("algo")
    assert monitor.is_enabled("algo")
    assert any("manually" in msg for msg in alerts)
