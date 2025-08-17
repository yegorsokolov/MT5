from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Tuple

import numpy as np

# ``utils.alerting`` pulls in a number of optional dependencies. Import lazily
# so tests don't require the full stack.
try:  # pragma: no cover - fallback when optional deps missing
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - fallback for tests
    def send_alert(msg: str) -> None:  # type: ignore
        pass


class PerformanceMonitor:
    """Track rolling Sharpe ratio and win-rate for algorithms.

    The monitor keeps a rolling window of trade returns for each algorithm and
    automatically disables strategies whose performance deteriorates beyond
    configured thresholds.  Disabled strategies remain inactive for a cooldown
    period before being reconsidered.  Operators are notified of disable and
    enable events via the :mod:`utils.alerting` module.

    Parameters
    ----------
    window:
        Number of recent trades to include in metric calculations.
    sharpe_threshold:
        Minimum rolling Sharpe ratio before a strategy is disabled.
    win_rate_threshold:
        Minimum proportion of winning trades before a strategy is disabled.
    cooldown:
        Time in seconds that a strategy remains disabled before it can be
        automatically re-enabled.
    clock:
        Function returning the current time as seconds since the epoch. Tests
        can inject a deterministic clock.
    alert_func:
        Function used to notify operators when strategies are disabled or
        enabled. Defaults to :func:`utils.alerting.send_alert`.
    """

    def __init__(
        self,
        window: int = 100,
        sharpe_threshold: float = 0.0,
        win_rate_threshold: float = 0.5,
        cooldown: float = 300.0,
        clock: Callable[[], float] = time.time,
        alert_func: Callable[[str], None] = send_alert,
    ) -> None:
        self.window = window
        self.sharpe_threshold = sharpe_threshold
        self.win_rate_threshold = win_rate_threshold
        self.cooldown = cooldown
        self.clock = clock
        self.alert_func = alert_func

        self._returns: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self._disabled_until: Dict[str, float | None] = {}
        self._metrics: Dict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))

    # ------------------------------------------------------------------
    def record_trade(self, algorithm: str, pnl: float) -> None:
        """Record ``pnl`` for ``algorithm`` and update metrics."""
        returns = self._returns[algorithm]
        returns.append(pnl)
        arr = np.array(returns, dtype=float)
        if len(arr) > 1:
            sharpe = float(np.mean(arr) / (np.std(arr, ddof=1) + 1e-8) * np.sqrt(len(arr)))
        else:
            sharpe = 0.0
        win_rate = float(np.mean(arr > 0))
        self._metrics[algorithm] = (sharpe, win_rate)

        now = self.clock()
        if self.is_enabled(algorithm):
            if sharpe < self.sharpe_threshold or win_rate < self.win_rate_threshold:
                self._disabled_until[algorithm] = now + self.cooldown
                self.alert_func(
                    f"Strategy {algorithm} disabled: sharpe={sharpe:.2f}, win-rate={win_rate:.2f}"
                )
        else:
            disabled_until = self._disabled_until.get(algorithm)
            if disabled_until is not None and now >= disabled_until:
                if sharpe >= self.sharpe_threshold and win_rate >= self.win_rate_threshold:
                    self._disabled_until[algorithm] = None
                    self.alert_func(
                        f"Strategy {algorithm} re-enabled: sharpe={sharpe:.2f}, win-rate={win_rate:.2f}"
                    )
                else:
                    # extend cooldown until metrics recover
                    self._disabled_until[algorithm] = now + self.cooldown

    # ------------------------------------------------------------------
    def is_enabled(self, algorithm: str) -> bool:
        """Return ``True`` if ``algorithm`` is currently enabled."""
        disabled_until = self._disabled_until.get(algorithm)
        if disabled_until is None:
            return True
        now = self.clock()
        if now >= disabled_until:
            sharpe, win_rate = self._metrics.get(algorithm, (0.0, 0.0))
            if sharpe >= self.sharpe_threshold and win_rate >= self.win_rate_threshold:
                self._disabled_until[algorithm] = None
                self.alert_func(
                    f"Strategy {algorithm} re-enabled: sharpe={sharpe:.2f}, win-rate={win_rate:.2f}"
                )
                return True
            # extend cooldown if still underperforming
            self._disabled_until[algorithm] = now + self.cooldown
        return False

    # ------------------------------------------------------------------
    def override_enable(self, algorithm: str) -> None:
        """Manually re-enable ``algorithm`` irrespective of metrics."""
        self._disabled_until[algorithm] = None
        self.alert_func(f"Strategy {algorithm} manually re-enabled")


__all__ = ["PerformanceMonitor"]
