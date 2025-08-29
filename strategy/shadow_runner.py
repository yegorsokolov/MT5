from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict

import numpy as np

from signal_queue import get_async_subscriber


@dataclass
class ShadowRunner:
    """Run a strategy in shadow mode and log hypothetical performance.

    The runner subscribes to the same signal feed as live strategies but does
    not place any trades.  Instead it invokes ``handler`` for each message to
    obtain a hypothetical PnL value.  Rolling performance metrics such as
    drawdown and Sharpe ratio are appended to ``reports/shadow/<name>.csv``.
    """

    name: str
    handler: Callable[[Dict[str, Any]], float]
    url: str | None = None
    window: int = 100
    out_dir: Path = Path("reports/shadow")
    _returns: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    async def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.name}.csv"
        equity = 0.0
        peak = 0.0
        async with get_async_subscriber(self.url) as sub:
            while True:
                msg = await sub.recv_json()
                pnl = float(self.handler(msg))
                self._returns.append(pnl)
                equity += pnl
                peak = max(peak, equity)
                drawdown = peak - equity
                sharpe = 0.0
                if len(self._returns) > 1:
                    arr = np.array(self._returns, dtype=float)
                    sharpe = float(np.mean(arr) / (np.std(arr, ddof=1) + 1e-8))
                with path.open("a") as f:
                    f.write(
                        f"{msg.get('Timestamp')},{pnl:.6f},{equity:.6f},{drawdown:.6f},{sharpe:.6f}\n"
                    )


__all__ = ["ShadowRunner"]
