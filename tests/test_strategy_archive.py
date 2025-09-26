from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy.archive import StrategyArchive


def test_strategy_archive_marks_important(tmp_path):
    (tmp_path / ".git").mkdir()
    archive = StrategyArchive(root=tmp_path / "reports" / "strategies", risk_loader=lambda: {"daily_drawdown": 0.05})

    entry = archive.record({"name": "alpha", "pnl": 0.2}, metadata={"monthly_profit": 0.2})
    assert entry.exists()
    important_files = list(archive.important_dir.glob("*.json"))
    assert any(path.name == entry.name for path in important_files)

    entries = list(archive.iter_entries())
    assert entries and entries[0]["important"] is True


def test_strategy_archive_handles_non_important(tmp_path):
    archive = StrategyArchive(root=tmp_path / "reports" / "strategies", risk_loader=lambda: {"daily_drawdown": 20.0})
    entry = archive.record({"name": "beta", "pnl": 0.1}, metadata={"monthly_profit": 0.1})
    assert not (archive.important_dir / entry.name).exists()
