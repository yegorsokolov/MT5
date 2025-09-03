"""Replay historical decisions and candidate strategies on capability changes.

This module watches :class:`utils.resource_monitor.ResourceMonitor` for
capability tier events.  When the local hardware tier is re-probed or upgraded
we reload newly enabled model variants and replay historical decisions through
them.  Candidate trading strategies can also be supplied and are evaluated
against the encrypted decision log to compute counterfactual PnL and risk
metrics.  Comparison reports are written to ``reports/replays`` and
``reports/strategy_replay`` with results fed back into the tournament
scoreboard for the :class:`strategy.router.StrategyRouter`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import logging

from analytics.metrics_store import record_metric

try:  # optional if backend not configured
    from core import state_sync
except Exception:  # pragma: no cover - optional dependency
    state_sync = None

logger = logging.getLogger(__name__)

REPLAY_DIR = Path(__file__).resolve().parent.parent / "reports" / "replays"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_REPLAY_DIR = (
    Path(__file__).resolve().parent.parent / "reports" / "strategy_replay"
)
STRATEGY_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

NEWS_REPLAY_DIR = (
    Path(__file__).resolve().parent.parent / "reports" / "news_replay"
)
NEWS_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

def reprocess_trades() -> None:
    """Re-run logged decisions through currently selected models."""
    from log_utils import read_decisions
    from state_manager import load_replay_timestamp, save_replay_timestamp
    from model_registry import select_models
    from analysis.replay_trades import replay_trades
    decisions = read_decisions()
    if decisions.empty:
        return
    last_ts = load_replay_timestamp()
    if last_ts:
        decisions = decisions[decisions["timestamp"] > pd.to_datetime(last_ts)]
    if decisions.empty:
        return
    versions = select_models()
    replay_trades(versions)
    save_replay_timestamp(decisions["timestamp"].max().isoformat())




def _risk_metrics(returns: Iterable[float]) -> Dict[str, float]:
    """Compute risk metrics for a return series.

    Parameters
    ----------
    returns:
        Iterable of periodic returns or PnL values.

    Returns
    -------
    dict
        Dictionary containing PnL, Sharpe ratio, max drawdown and a simple
        tail-risk proxy (CVaR at 5%).
    """

    arr = pd.Series(list(returns), dtype=float)
    if arr.empty:
        return {
            "pnl": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0,
            "cvar": 0.0,
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    sharpe = mean / (std + 1e-9)
    cumulative = (1 + arr).cumprod()
    drawdown = float((cumulative.cummax() - cumulative).max())
    # 5% CVaR (expected shortfall)
    var_threshold = arr.quantile(0.05)
    tail = arr[arr <= var_threshold]
    cvar = -float(tail.mean()) if not tail.empty else 0.0
    return {
        "pnl": float(arr.sum()),
        "sharpe": sharpe,
        "drawdown": drawdown,
        "cvar": cvar,
    }


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
    """Reprocess historical trades applying the latest news impact model.

    Generates comparison reports including per-trade PnL deltas and Sharpe
    ratio differences.  ``out_dir`` defaults to ``reports/replays``.
    """
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
    risk_dir = out_dir.parent / "replay_risk"
    risk_dir.mkdir(parents=True, exist_ok=True)
    adj = []
    for row in trades.itertuples():
        impact, _ = impact_model.get_impact(row.symbol, row.timestamp)
        adj.append(row.pnl * (1 + (impact or 0)))
    trades["reprocessed_pnl"] = adj
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        trades.to_parquet(out_dir / "reprocessed.parquet", index=False)
    except Exception:
        trades.to_csv(out_dir / "reprocessed.csv", index=False)

    comp = trades[["timestamp", "pnl", "reprocessed_pnl"]].rename(
        columns={"pnl": "pnl_old", "reprocessed_pnl": "pnl_new"}
    )
    comp["pnl_delta"] = comp["pnl_new"] - comp["pnl_old"]
    comp.to_csv(out_dir / "old_vs_new_pnl.csv", index=False)

    old_metrics = _risk_metrics(comp["pnl_old"])
    new_metrics = _risk_metrics(comp["pnl_new"])

    sharpe_delta = new_metrics["sharpe"] - old_metrics["sharpe"]
    drawdown_delta = new_metrics["drawdown"] - old_metrics["drawdown"]
    cvar_delta = new_metrics["cvar"] - old_metrics["cvar"]

    pd.DataFrame(
        [
            {
                "sharpe_old": old_metrics["sharpe"],
                "sharpe_new": new_metrics["sharpe"],
                "sharpe_delta": sharpe_delta,
            }
        ]
    ).to_csv(out_dir / "sharpe_deltas.csv", index=False)

    summary = pd.DataFrame(
        {
            "pnl_old": [float(comp["pnl_old"].sum())],
            "pnl_new": [float(comp["pnl_new"].sum())],
            "sharpe_old": [old_metrics["sharpe"]],
            "sharpe_new": [new_metrics["sharpe"]],
            "sharpe_delta": [sharpe_delta],
            "drawdown_old": [old_metrics["drawdown"]],
            "drawdown_new": [new_metrics["drawdown"]],
            "drawdown_delta": [drawdown_delta],
            "cvar_old": [old_metrics["cvar"]],
            "cvar_new": [new_metrics["cvar"]],
            "cvar_delta": [cvar_delta],
        }
    )
    summary_path = out_dir / "pnl_summary.parquet"
    try:
        summary.to_parquet(summary_path, index=False)
    except Exception:
        summary.to_csv(out_dir / "pnl_summary.csv", index=False)

    # side-by-side risk comparison output
    risk_comp = pd.DataFrame(
        [
            {"metric": "pnl", "original": summary["pnl_old"].iloc[0], "replay": summary["pnl_new"].iloc[0], "delta": summary["pnl_new"].iloc[0] - summary["pnl_old"].iloc[0]},
            {"metric": "sharpe", "original": old_metrics["sharpe"], "replay": new_metrics["sharpe"], "delta": sharpe_delta},
            {"metric": "drawdown", "original": old_metrics["drawdown"], "replay": new_metrics["drawdown"], "delta": drawdown_delta},
            {"metric": "cvar", "original": old_metrics["cvar"], "replay": new_metrics["cvar"], "delta": cvar_delta},
        ]
    )
    try:
        risk_comp.to_parquet(risk_dir / "risk_comparison.parquet", index=False)
    except Exception:
        risk_comp.to_csv(risk_dir / "risk_comparison.csv", index=False)

    try:
        record_metric("replay_pnl_old", summary["pnl_old"].iloc[0])
        record_metric("replay_pnl_new", summary["pnl_new"].iloc[0])
        record_metric("replay_pnl_diff", summary["pnl_new"].iloc[0] - summary["pnl_old"].iloc[0])
    except Exception:
        pass
    return summary


def news_replay(
    trade_path: Path | None = None,
    news_cache_dir: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Replay trades with and without archived news features.

    This loads sentiment/impact vectors from ``data/news_cache`` and merges
    them with the recorded trade PnL.  Profitability is then compared with the
    news features included versus ignored.  Comparison reports are written to
    ``reports/news_replay`` by default.
    """

    trade_path = Path(trade_path) if trade_path else Path("reports/trades.csv")
    if not trade_path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(trade_path, parse_dates=["timestamp"])

    vectors = pd.DataFrame()
    cache_dir = Path(news_cache_dir) if news_cache_dir else Path("data/news_cache")
    parquet_path = cache_dir / "news_features.parquet"
    json_path = cache_dir / "stock_headlines.json"
    if parquet_path.exists():
        try:
            vectors = pd.read_parquet(parquet_path)
        except Exception:
            vectors = pd.DataFrame()
    elif json_path.exists():
        try:
            raw = pd.read_json(json_path)
            vectors = raw[["symbol", "timestamp", "sentiment"]]
            vectors["timestamp"] = pd.to_datetime(vectors["timestamp"])
        except Exception:
            vectors = pd.DataFrame()

    if vectors.empty:
        trades["sentiment"] = 0.0
    else:
        trades = trades.merge(
            vectors[["symbol", "timestamp", "sentiment"]],
            on=["symbol", "timestamp"],
            how="left",
        )
        trades["sentiment"] = trades["sentiment"].fillna(0.0)

    trades["pnl_without_news"] = trades["pnl"]
    trades["pnl_with_news"] = trades["pnl"] * (1 + trades["sentiment"])
    trades["pnl_delta"] = trades["pnl_with_news"] - trades["pnl_without_news"]

    out_dir = Path(out_dir) if out_dir else NEWS_REPLAY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    trades[
        ["timestamp", "symbol", "pnl_without_news", "pnl_with_news", "pnl_delta"]
    ].to_csv(out_dir / "trade_comparison.csv", index=False)

    old_metrics = _risk_metrics(trades["pnl_without_news"])
    new_metrics = _risk_metrics(trades["pnl_with_news"])
    summary = pd.DataFrame(
        {
            "pnl_without_news": [old_metrics["pnl"]],
            "pnl_with_news": [new_metrics["pnl"]],
            "pnl_delta": [new_metrics["pnl"] - old_metrics["pnl"]],
            "sharpe_without_news": [old_metrics["sharpe"]],
            "sharpe_with_news": [new_metrics["sharpe"]],
            "sharpe_delta": [new_metrics["sharpe"] - old_metrics["sharpe"]],
        }
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
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
    from features import report_status

    monitor.start()
    registry = ModelRegistry(auto_refresh=False)
    queue = monitor.subscribe()
    prev_models: Dict[str, str] = {
        task: registry.get(task) for task in registry.selected
    }
    status = report_status()
    prev_features = {
        f["name"]
        for f in status.get("features", [])
        if f.get("status") == "active"
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
        status = report_status()
        active_features = {
            f["name"]
            for f in status.get("features", [])
            if f.get("status") == "active"
        }
        new_features = active_features - prev_features
        new_models: List[str] = [
            m for t, m in current.items() if prev_models.get(t) != m
        ]
        if new_models or new_features:
            from .replay_trades import replay_trades

            replay_trades(list(current.values()))
            _flag_discrepancies(threshold)
            summary = reprocess(REPLAY_DIR)
            logger.info(
                "Active features after upgrade: %s", sorted(active_features)
            )
            try:
                record_metric("active_feature_count", len(active_features))
                if not summary.empty:
                    record_metric("replay_sharpe_new", float(summary["sharpe_new"].iloc[0]))
            except Exception:
                pass

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
        prev_features = active_features


def main() -> None:  # pragma: no cover - CLI entry
    asyncio.run(watch_upgrades())


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
