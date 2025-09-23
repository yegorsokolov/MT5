import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt5.risk_manager import RiskManager


def test_combined_drawdown_triggers_global_stop() -> None:
    rm = RiskManager(max_drawdown=100)

    async def bot(loss: float) -> None:
        rm.update("bot", loss)

    asyncio.get_event_loop().run_until_complete(
        asyncio.gather(bot(-60), bot(-50))
    )
    assert rm.status()["trading_halted"] is True


def test_check_fills_returns_violations() -> None:
    rm = RiskManager(max_drawdown=100)
    violations = rm.check_fills(
        placed=10,
        filled=4,
        cancels=6,
        slippage=0.02,
        min_ratio=0.5,
        max_slippage=0.01,
        max_cancel_rate=0.5,
    )
    assert set(violations) == {"fill_ratio", "cancel_rate", "slippage"}
