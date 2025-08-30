"""Replay historical decisions and candidate strategies on capability changes.

This module watches :class:`utils.resource_monitor.ResourceMonitor` for
capability tier events.  When the local hardware tier is re-probed or upgraded
we reload newly enabled model variants and replay historical decisions through
them.  Candidate trading strategies can also be supplied and are evaluated
against the encrypted decision log to compute counterfactual PnL and risk
metrics.  Comparison reports are written to ``reports/replay`` and
``reports/strategy_replay`` with results fed back into the tournament
scoreboard for the :class:`strategy.router.StrategyRouter`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


from analytics.metrics_store import record_metric

try:  # optional if backend not configured
    from core import state_sync
except Exception:  # pragma: no cover - optional dependency
    state_sync = None

REPLAY_DIR = Path(__file__).resolve().parent.parent / "reports" / "replay"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_REPLAY_DIR = (
    Path(__file__).resolve().parent.parent / "reports" / "strategy_replay"
)
STRATEGY_REPLAY_DIR.mkdir(parents=True, exist_ok=True)


def _risk_metrics(returns: Iterable[float]) -> Dict[str, float]:
    """Compute simple risk metrics for a return series."""
    arr = pd.Series(list(returns), dtype=float)
    if arr.empty:
        return {"pnl": 0.0, "sharpe": 0.0, "drawdown": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    sharpe = mean / (std + 1e-9)
    cumulative = (1 + arr).cumprod()
    drawdown = float((cumulative.cummax() - cumulative).max())
    return {"pnl": float(arr.sum()), "sharpe": sharpe, "drawdown": drawdown}


def replay_strategies(strategy_ids: List[str]) -> pd.DataFrame:
    """Replay candidate strategies over logged decisions.

    Parameters
    ----------
    strategy_ids:
        List of strategy identifiers to evaluate.
    """

    if not strategy_ids:
        return pd.DataFrame()

    from log_utils import read_decisions

    decisions = read_decisions()
    if decisions.empty or "return" not in decisions.columns:
        return pd.DataFrame()

    from strategy.router import StrategyRouter

    router = StrategyRouter()
    features = ["volatility", "trend_strength", "regime", "market_basket"]
    for col in features:
        if col not in decisions.columns:
            decisions[col] = 0.0

    summary_rows: List[Dict[str, float]] = []
    scoreboard_rows: List[Dict[str, float]] = []

    for sid in strategy_ids:
        algo = router.algorithms.get(sid)
        if algo is None:
            continue
        acts = [
            algo(
                {
                    "volatility": row.volatility,
                    "trend_strength": row.trend_strength,
                    "regime": row.regime,
                    "market_basket": row.market_basket,
                }
            )
            for row in decisions[features].itertuples(index=False)
        ]
        pnl = pd.Series(acts) * decisions["return"].to_numpy()
        comp = pd.DataFrame(
            {
                "timestamp": decisions["timestamp"],
                "return": decisions["return"],
                "action": acts,
                "pnl": pnl,
                "regime": decisions.get("regime", 0),
                "market_basket": decisions.get("market_basket", 0),
            }
        )
        try:
            comp.to_parquet(STRATEGY_REPLAY_DIR / f"{sid}.parquet", index=False)
        except Exception:
            comp.to_csv(STRATEGY_REPLAY_DIR / f"{sid}.csv", index=False)
        metrics = _risk_metrics(pnl)
        summary_rows.append({"algorithm": sid, **metrics})
        for basket, df_basket in comp.groupby("market_basket"):
            rm = _risk_metrics(df_basket["pnl"])
            scoreboard_rows.append({"market_basket": basket, "algorithm": sid, **rm})

    if not summary_rows:
        return pd.DataFrame()

    summary = pd.DataFrame(summary_rows)
    try:
        summary.to_parquet(STRATEGY_REPLAY_DIR / "summary.parquet", index=False)
    except Exception:
        summary.to_csv(STRATEGY_REPLAY_DIR / "summary.csv", index=False)

    # Update tournament scoreboard
    router = StrategyRouter()
    path = router.scoreboard_path
    try:
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    except Exception:
        existing = pd.DataFrame()
    new_scores = pd.DataFrame(scoreboard_rows).set_index(["market_basket", "algorithm"])
    if not existing.empty:
        existing = existing.reset_index()
        combined = pd.concat([existing, new_scores.reset_index()])
    else:
        combined = new_scores.reset_index()
    combined = combined.drop_duplicates(
        subset=["market_basket", "algorithm"], keep="last"
    )
    combined = combined.set_index(["market_basket", "algorithm"])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path)
    except Exception:
        pass

    return summary


def reprocess(out_dir: Path | None = None) -> pd.DataFrame:
    """Reprocess historical trades applying the latest news impact model."""
    import importlib.util

    news_dir = Path(__file__).resolve().parent.parent / "news" / "impact_model.py"
    spec = importlib.util.spec_from_file_location("news.impact_model", news_dir)
    impact_model = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader  # for type checkers
    spec.loader.exec_module(impact_model)

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
    """Monitor capability tier events and trigger decision/strategy replays."""

    from model_registry import ModelRegistry
    from utils.resource_monitor import monitor

    monitor.start()
    registry = ModelRegistry(auto_refresh=False)
    queue = monitor.subscribe()
    prev_models: Dict[str, str] = {
        task: registry.get(task) for task in registry.selected
    }

    while True:
        try:
            await asyncio.wait_for(queue.get(), timeout=24 * 60 * 60)
        except asyncio.TimeoutError:
            pass
        registry.refresh()
        current: Dict[str, str] = {
            task: registry.get(task) for task in registry.selected
        }
        new_models: List[str] = [
            m for t, m in current.items() if prev_models.get(t) != m
        ]
        if new_models:
            from .replay_trades import replay_trades

            replay_trades(new_models)
            _flag_discrepancies(threshold)
            reprocess(REPLAY_DIR)

        # Check for newly synthesised strategies
        from strategy.router import StrategyRouter

        router = StrategyRouter()
        try:
            sb = (
                pd.read_parquet(router.scoreboard_path)
                if router.scoreboard_path.exists()
                else pd.DataFrame()
            )
            seen = (
                sb.index.get_level_values("algorithm").unique().tolist()
                if not sb.empty
                else []
            )
        except Exception:
            seen = []
        candidates = [name for name in router.algorithms if name not in seen]
        if candidates:
            replay_strategies(candidates)

        prev_models = current


def main() -> None:  # pragma: no cover - CLI entry
    asyncio.run(watch_upgrades())


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
