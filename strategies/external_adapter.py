"""Adapter for running external Freqtrade or Backtrader strategies."""

from __future__ import annotations

import importlib.util
import pandas as pd
from pathlib import Path

try:
    from freqtrade.strategy import IStrategy  # type: ignore
except Exception:  # pragma: no cover - optional dep
    IStrategy = None  # type: ignore

try:
    import backtrader as bt  # type: ignore
except Exception:  # pragma: no cover - optional dep
    bt = None  # type: ignore


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("ext_strategy", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load strategy at {path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_strategy(path: str) -> tuple[str, type]:
    """Return (framework, class) tuple for a strategy file."""
    module = _load_module(path)
    strategy_cls = None
    framework = None
    for obj in module.__dict__.values():
        if IStrategy and isinstance(obj, type) and issubclass(obj, IStrategy) and obj is not IStrategy:
            strategy_cls = obj
            framework = "freqtrade"
            break
        if bt and isinstance(obj, type) and hasattr(bt, "Strategy") and issubclass(obj, bt.Strategy) and obj is not bt.Strategy:
            strategy_cls = obj
            framework = "backtrader"
            break
    if not strategy_cls or not framework:
        raise ValueError("No strategy class found in file")
    return framework, strategy_cls


def _compute_metrics(returns: pd.Series) -> dict:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    sharpe = (returns.mean() / returns.std(ddof=0)) * (len(returns) ** 0.5) if len(returns) > 1 else 0.0
    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min() * 100) if not drawdown.empty else 0.0,
        "total_return": float(cumulative.iloc[-1] - 1) if not cumulative.empty else 0.0,
        "win_rate": float((returns > 0).mean() * 100) if not returns.empty else 0.0,
    }


def _run_freqtrade(df: pd.DataFrame, cls: type) -> dict:
    if IStrategy is None:
        raise ImportError("freqtrade is not installed")
    strat = cls()
    ft_df = df.copy()
    price = ft_df.get("mid") or ft_df.iloc[:, 0]
    ft_df["close"] = price
    ft_df["open"] = price
    ft_df["high"] = price
    ft_df["low"] = price
    ft_df["volume"] = ft_df.get("Volume", 0)
    ft_df = strat.populate_indicators(ft_df, {})
    ft_df = strat.populate_entry_trend(ft_df, {})
    ft_df = strat.populate_exit_trend(ft_df, {})
    returns = []
    in_pos = False
    entry = 0.0
    for _, row in ft_df.iterrows():
        if not in_pos and row.get("enter_long"):
            in_pos = True
            entry = row["close"]
            continue
        if in_pos and row.get("exit_long"):
            returns.append((row["close"] - entry) / entry)
            in_pos = False
    return _compute_metrics(pd.Series(returns))


def _run_backtrader(df: pd.DataFrame, cls: type) -> dict:
    if bt is None:
        raise ImportError("backtrader is not installed")

    class Feed(bt.feeds.PandasData):
        params = dict(
            datetime="Timestamp",
            open="mid",
            high="mid",
            low="mid",
            close="mid",
            volume=None,
            openinterest=None,
        )

    cerebro = bt.Cerebro()
    cerebro.addstrategy(cls)
    cerebro.adddata(Feed(dataname=df))
    cerebro.broker.setcash(10000.0)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    result = cerebro.run()
    strat = result[0]
    ret_series = pd.Series(strat.analyzers.timereturn.get_analysis())
    metrics = _compute_metrics(ret_series)

    trade_data = strat.analyzers.trades.get_analysis()
    total = trade_data.get("total", {}).get("closed", 0)
    won = trade_data.get("won", {}).get("total", 0)
    if total:
        metrics["win_rate"] = won / total * 100.0
    dd = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0)
    metrics["max_drawdown"] = dd
    metrics["total_return"] = (cerebro.broker.getvalue() - 10000.0) / 10000.0
    return metrics


def run_external_strategy(df: pd.DataFrame, path: str) -> dict:
    """Execute an external strategy against ``df`` and return metrics."""
    framework, cls = load_strategy(str(path))
    if framework == "freqtrade":
        return _run_freqtrade(df, cls)
    if framework == "backtrader":
        return _run_backtrader(df, cls)
    raise ValueError(f"Unknown framework {framework}")
