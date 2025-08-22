"""Replay historical decisions when resource capabilities improve.

This module watches :class:`utils.resource_monitor.ResourceMonitor` for
capability tier upgrades.  When the local hardware tier increases we reload
newly enabled model variants and replay historical decisions through them.  A
comparison report is written to ``reports/replay`` and discrepancies are flagged
for manual review.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

from analytics.metrics_store import record_metric
from news import impact_model
from model_registry import ModelRegistry, TIERS
from utils.resource_monitor import monitor
from .replay_trades import replay_trades

try:  # optional if backend not configured
    from core import state_sync
except Exception:  # pragma: no cover - optional dependency
    state_sync = None

REPLAY_DIR = Path(__file__).resolve().parent.parent / "reports" / "replay"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)


def reprocess(out_dir: Path | None = None) -> pd.DataFrame:
    """Reprocess historical trades applying the latest news impact model."""
    trade_path = Path("reports/trades.csv")
    if state_sync:
        try:
            state_sync.pull_decisions()
        except Exception:
            pass
    if not trade_path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(trade_path, parse_dates=["timestamp"])
    out_dir = Path(out_dir) if out_dir else REPLAY_DIR
    adj = []
    for row in trades.itertuples():
        impact, _ = impact_model.get_impact(row.symbol, row.timestamp)
        adj.append(row.pnl * (1 + (impact or 0)))
    trades["reprocessed_pnl"] = adj
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(out_dir / "reprocessed.parquet", index=False)
    summary = pd.DataFrame(
        {
            "pnl_old": [float(trades["pnl"].sum())],
            "pnl_new": [float(trades["reprocessed_pnl"].sum())],
        }
    )
    summary_path = out_dir / "pnl_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    try:
        record_metric("replay_pnl_old", summary["pnl_old"].iloc[0])
        record_metric("replay_pnl_new", summary["pnl_new"].iloc[0])
        record_metric(
            "replay_pnl_diff",
            summary["pnl_new"].iloc[0] - summary["pnl_old"].iloc[0],
        )
    except Exception:
        pass
    return summary


def _flag_discrepancies(threshold: float = 0.05) -> Dict[str, float]:
    """Read replay summary and flag large deviations."""

    flagged: Dict[str, float] = {}
    summary_path = REPLAY_DIR / "summary.parquet"
    if not summary_path.exists():
        return flagged
    try:
        df = pd.read_parquet(summary_path)
        flagged = {
            row.version: float(row.mae)
            for row in df.itertuples()
            if row.mae > threshold
        }
        (REPLAY_DIR / "latest.json").write_text(
            json.dumps({"flagged": flagged})
        )
    except Exception:
        pass
    return flagged


async def watch_upgrades(threshold: float = 0.05) -> None:
    """Monitor capability tier upgrades and trigger decision replays."""

    monitor.start()
    registry = ModelRegistry(auto_refresh=False)
    queue = monitor.subscribe()
    prev_tier = monitor.capability_tier
    prev_models: Dict[str, str] = {
        task: registry.get(task) for task in registry.selected
    }
    while True:
        tier = await queue.get()
        if TIERS.get(tier, 0) > TIERS.get(prev_tier, 0):
            registry.refresh()
            current: Dict[str, str] = {
                task: registry.get(task) for task in registry.selected
            }
            new_models: List[str] = [
                m for t, m in current.items() if prev_models.get(t) != m
            ]
            if new_models:
                replay_trades(new_models)
                _flag_discrepancies(threshold)
                reprocess(REPLAY_DIR)
        prev_models = {
            task: registry.get(task) for task in registry.selected
        }
        prev_tier = tier


def main() -> None:  # pragma: no cover - CLI entry
    asyncio.run(watch_upgrades())


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
