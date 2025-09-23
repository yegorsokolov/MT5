"""Simple backtesting for the Adaptive MT5 bot."""

import logging
import asyncio
from pathlib import Path
import mt5.log_utils as log_utils
from mt5.log_utils import setup_logging, log_exceptions
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from data.feature_scaler import FeatureScaler

from utils import load_config
from analysis.strategy_evaluator import StrategyEvaluator
from execution import ExecutionEngine
from utils.resource_monitor import monitor
from data.history import load_history_parquet, load_history_config

try:  # optional simulator for offline experiments
    from simulation.agent_market import AgentMarketSimulator
except Exception:  # pragma: no cover - simulator optional
    AgentMarketSimulator = None  # type: ignore
from data.features import make_features
from mt5.ray_utils import ray, init as ray_init, shutdown as ray_shutdown

try:  # optional metrics integration
    from mt5.metrics import (
        SLIPPAGE_BPS,
        PARTIAL_FILL_COUNT,
        SKIPPED_TRADE_COUNT,
    )
except Exception:  # pragma: no cover - metrics are optional
    SLIPPAGE_BPS = PARTIAL_FILL_COUNT = SKIPPED_TRADE_COUNT = None

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for backtest routines."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


LOG_DIR = getattr(log_utils, "LOG_DIR", Path(__file__).resolve().parents[1] / "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
BACKTEST_STATS = LOG_DIR / "backtest_stats.csv"


BASE_FEATURES = [
    "return",
    "ma_5",
    "ma_10",
    "ma_30",
    "ma_60",
    "volatility_30",
    "spread",
    "rsi_14",
    "news_sentiment",
]


def feature_columns(df: pd.DataFrame) -> list:
    cols = [c for c in BASE_FEATURES if c in df.columns]
    cols.extend(
        [
            c
            for c in df.columns
            if c.startswith("cross_corr_")
            or c.startswith("factor_")
            or c.startswith("cross_mom_")
        ]
    )
    if "volume_ratio" in df.columns:
        cols.extend(["volume_ratio", "volume_imbalance"])
    if "SymbolCode" in df.columns:
        cols.append("SymbolCode")
    return cols


def trailing_stop(
    entry_price: float, current_price: float, stop: float, distance: float
) -> float:
    """Update trailing stop based on price movement."""
    if current_price - distance > stop:
        return current_price - distance
    return stop


def compute_metrics(returns: pd.Series) -> dict:
    if returns.empty:
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "return": 0.0,
            "win_rate": 0.0,
        }

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    drawdown_min = drawdown.min()
    std = returns.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float(np.sqrt(252) * returns.mean() / std)

    max_drawdown = float(drawdown_min) if np.isfinite(drawdown_min) else 0.0
    total_return = float(cumulative.iloc[-1] - 1)
    win_rate = float((returns > 0).mean()) * 100
    if not np.isfinite(win_rate):
        win_rate = 0.0

    metrics = {
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown * 100,
        "total_return": total_return,
        "win_rate": win_rate,
    }
    metrics["return"] = metrics["total_return"]
    return metrics


def bootstrap_sharpe_pvalue(
    returns: pd.Series, *, n_bootstrap: int = 1000, seed: int = 42
) -> float:
    """Estimate a p-value for the Sharpe ratio using bootstrap resampling."""
    if returns.empty or returns.std(ddof=0) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = np.sqrt(252) * returns.mean() / returns.std(ddof=0)
    centered = returns - returns.mean()
    sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(centered, size=len(centered), replace=True)
        if sample.std(ddof=0) == 0:
            continue
        sharpe = np.sqrt(252) * sample.mean() / sample.std(ddof=0)
        sharpes.append(sharpe)
    if not sharpes:
        return float("nan")
    sharpes = np.array(sharpes)
    if observed > 0:
        return float(np.mean(sharpes >= observed))
    return float(np.mean(sharpes <= observed))


def log_backtest_stats(metrics: dict) -> None:
    """Append backtest metrics to ``logs/backtest_stats.csv``."""
    df = pd.DataFrame([metrics])
    df.to_csv(BACKTEST_STATS, mode="a", header=not BACKTEST_STATS.exists(), index=False)


def fit_model(train_df: pd.DataFrame, cfg: dict) -> Pipeline:
    """Train a LightGBM model on the given dataframe."""
    feats = feature_columns(train_df)
    X = train_df[feats]
    y = (train_df["return"].shift(-1) > 0).astype(int)
    steps = []
    if cfg.get("use_scaler", True):
        steps.append(("scaler", FeatureScaler()))
    steps.append(("clf", LGBMClassifier(n_estimators=200, random_state=42)))
    pipe = Pipeline(steps)
    pipe.fit(X, y)
    return pipe


def backtest_on_df(
    df: pd.DataFrame,
    model,
    cfg: dict,
    *,
    slippage_model=None,
    latency_ms: int = 0,
    return_returns: bool = False,
) -> dict | tuple[dict, pd.Series]:
    """Run the trading simulation on a dataframe using the given model.

    Parameters
    ----------
    df: pd.DataFrame
        Price history with engineered features.
    model: Any
        Trained sklearn-compatible model used to generate probabilities.
    cfg: dict
        Configuration with threshold and trailing stop settings.
    slippage_model: Callable, optional
        When provided, ``ExecutionEngine`` will query this callable with
        ``(order_size, side)`` to estimate slippage in basis points.
    latency_ms: int, optional
        Artificial execution latency in milliseconds.  Orders are filled
        using market data delayed by this amount.
    return_returns: bool, optional
        When ``True`` the list of trade returns is also returned for
        statistical testing.
    """
    feats = feature_columns(df)
    probs = model.predict_proba(df[feats])[:, 1]

    threshold = cfg.get("threshold", 0.55)
    regime_thresholds = getattr(model, "regime_thresholds", {})
    distance = cfg.get("trailing_stop_pips", 20) * 1e-4
    order_size = cfg.get("order_size", 1.0)

    engine = ExecutionEngine()
    try:
        engine.start_optimizer()
        tier = getattr(monitor, "capability_tier", "lite")
        strategy = "ioc" if tier == "lite" else cfg.get("execution_strategy", "vwap")

        # Pre-compute indices for delayed market data if latency is specified
        if latency_ms > 0 and "Timestamp" in df.columns:
            ts = pd.to_datetime(df["Timestamp"]).to_numpy()
            delays = ts + np.timedelta64(int(latency_ms), "ms")
            delay_idx = np.searchsorted(ts, delays, side="left")
            delay_idx = np.clip(delay_idx, 0, len(df) - 1)
        else:
            delay_idx = np.arange(len(df))

        async def _run_backtest() -> tuple[list[float], int, int]:
            in_position = False
            entry = 0.0
            stop = 0.0
            returns: list[float] = []
            skipped = 0
            partial = 0

            for i, (row, prob) in enumerate(zip(df.itertuples(index=False), probs)):
                delayed = df.iloc[delay_idx[i]]
                price_mid = getattr(delayed, "mid")
                bid = getattr(delayed, "Bid", price_mid)
                ask = getattr(delayed, "Ask", price_mid)
                bid_vol = getattr(delayed, "BidVolume", np.inf)
                ask_vol = getattr(delayed, "AskVolume", np.inf)

                engine.record_volume(bid_vol + ask_vol)
                regime = getattr(row, "market_regime", None)
                thr = (
                    regime_thresholds.get(int(regime), threshold)
                    if regime is not None
                    else threshold
                )
                if not in_position and prob > thr:
                    result = await engine.place_order(
                        side="buy",
                        quantity=order_size,
                        bid=bid,
                        ask=ask,
                        bid_vol=bid_vol,
                        ask_vol=ask_vol,
                        mid=price_mid,
                        strategy=strategy,
                        expected_slippage_bps=cfg.get("slippage_bps", 0.0),
                        slippage_model=slippage_model,
                    )
                    if result["filled"] < order_size:
                        skipped += 1
                        continue
                    in_position = True
                    entry = result["avg_price"]
                    stop = entry - distance
                    continue
                if in_position:
                    current_mid = getattr(row, "mid")
                    stop = trailing_stop(entry, current_mid, stop, distance)
                    if current_mid <= stop:
                        result = await engine.place_order(
                            side="sell",
                            quantity=order_size,
                            bid=bid,
                            ask=ask,
                            bid_vol=bid_vol,
                            ask_vol=ask_vol,
                            mid=price_mid,
                            strategy=strategy,
                            expected_slippage_bps=cfg.get("slippage_bps", 0.0),
                            slippage_model=slippage_model,
                        )
                        fill_frac = min(result["filled"] / order_size, 1.0)
                        if fill_frac < 1.0:
                            partial += 1
                        exit_price = result["avg_price"]
                        returns.append(((exit_price - entry) / entry) * fill_frac)
                        in_position = False

            return returns, skipped, partial

        returns, skipped_trades, partial_fills = asyncio.run(_run_backtest())
        series = pd.Series(returns)
        metrics = compute_metrics(series)
        metrics["trade_count"] = int(len(series))
        metrics.setdefault("return", metrics.get("total_return", 0.0))
        metrics["skipped_trades"] = skipped_trades
        metrics["partial_fills"] = partial_fills
        metrics["sharpe_p_value"] = bootstrap_sharpe_pvalue(series)
        if SLIPPAGE_BPS is not None:
            SLIPPAGE_BPS.set(cfg.get("slippage_bps", 0.0))
            PARTIAL_FILL_COUNT.inc(partial_fills)
            SKIPPED_TRADE_COUNT.inc(skipped_trades)
        log_backtest_stats(metrics)
        if return_returns:
            return metrics, series
        return metrics
    finally:
        engine.stop_optimizer()


def run_backtest(
    cfg: dict,
    *,
    return_returns: bool = False,
    external_strategy: str | None = None,
    latency_ms: int = 0,
    slippage_model=None,
) -> dict | tuple[dict, pd.Series]:
    init_logging()
    if cfg.get("sim_env") and AgentMarketSimulator is not None:
        sim = AgentMarketSimulator(
            seed=int(cfg.get("sim_seed", 42)),
            steps=int(cfg.get("sim_steps", 1000)),
        )
        _, book = sim.run()
        df = sim.to_history_df(book)
        df["Symbol"] = cfg.get("symbol", "SIM")
    else:
        data_path = Path(__file__).resolve().parent / "data" / "history.parquet"
        if not data_path.exists():
            cfg_root = Path(__file__).resolve().parent
            symbols = cfg.get("symbols") or [cfg.get("symbol")]
            dfs = []
            for sym in symbols:
                df_sym = load_history_config(sym, cfg, cfg_root)
                df_sym["Symbol"] = sym
                dfs.append(df_sym)
            df_all = pd.concat(dfs, ignore_index=True)
            save_path = data_path
            save_path.parent.mkdir(exist_ok=True)
            save_path.unlink(missing_ok=True)
            df_all.to_parquet(save_path, index=False)

        df = load_history_parquet(data_path)
        df = df[df.get("Symbol").isin([cfg.get("symbol")])]
    df = make_features(df)
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes
    if external_strategy:
        from strategies.external_adapter import run_external_strategy

        return run_external_strategy(df, external_strategy)

    model = joblib.load(Path(__file__).resolve().parent / "model.joblib")

    return backtest_on_df(
        df,
        model,
        cfg,
        slippage_model=slippage_model,
        latency_ms=latency_ms,
        return_returns=return_returns,
    )


@ray.remote
def _backtest_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    start: str,
    end: str,
    *,
    latency_ms: int = 0,
    slippage_model=None,
) -> dict:
    """Train a model on ``train_df`` and backtest on ``test_df``."""
    model = fit_model(train_df, cfg)
    metrics = backtest_on_df(
        test_df,
        model,
        cfg,
        slippage_model=slippage_model,
        latency_ms=latency_ms,
    )
    metrics["period_start"] = start
    metrics["period_end"] = end
    return metrics


def run_rolling_backtest(
    cfg: dict,
    external_strategy: str | None = None,
    *,
    latency_ms: int = 0,
    slippage_model=None,
) -> dict:
    """Perform rolling train/test backtests and aggregate metrics."""
    init_logging()
    if cfg.get("sim_env") and AgentMarketSimulator is not None:
        sim = AgentMarketSimulator(
            seed=int(cfg.get("sim_seed", 42)),
            steps=int(cfg.get("sim_steps", 1000)),
        )
        _, book = sim.run()
        df = sim.to_history_df(book)
        df["Symbol"] = cfg.get("symbol", "SIM")
    else:
        data_path = Path(__file__).resolve().parent / "data" / "history.parquet"
        if not data_path.exists():
            cfg_root = Path(__file__).resolve().parent
            symbols = cfg.get("symbols") or [cfg.get("symbol")]
            dfs = []
            for sym in symbols:
                df_sym = load_history_config(sym, cfg, cfg_root)
                df_sym["Symbol"] = sym
                dfs.append(df_sym)
            df_all = pd.concat(dfs, ignore_index=True)
            data_path.parent.mkdir(exist_ok=True)
            df_all.to_parquet(data_path, index=False)

        df = load_history_parquet(data_path)
        df = df[df.get("Symbol").isin([cfg.get("symbol")])]
    df = make_features(df)
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes
    df = df.sort_values("Timestamp").reset_index(drop=True)

    window = int(cfg.get("backtest_window_months", 6))
    step = int(cfg.get("backtest_step_months", window))

    start = df["Timestamp"].min()
    end = df["Timestamp"].max()

    parallel = cfg.get("parallel_backtest", False) and not external_strategy
    if parallel:
        ray_init(num_cpus=cfg.get("ray_num_cpus"))
        futures = []
    period_metrics = []
    while start + pd.DateOffset(months=window * 2) <= end:
        train_end = start + pd.DateOffset(months=window)
        test_end = train_end + pd.DateOffset(months=window)

        train_df = df[(df["Timestamp"] >= start) & (df["Timestamp"] < train_end)]
        test_df = df[(df["Timestamp"] >= train_end) & (df["Timestamp"] < test_end)]
        if train_df.empty or test_df.empty:
            start += pd.DateOffset(months=step)
            continue

        start_iso = train_end.date().isoformat()
        end_iso = test_end.date().isoformat()
        if parallel:
            futures.append(
                _backtest_window.remote(
                    train_df,
                    test_df,
                    cfg,
                    start_iso,
                    end_iso,
                    latency_ms=latency_ms,
                    slippage_model=slippage_model,
                )
            )
        else:
            if external_strategy:
                from strategies.external_adapter import run_external_strategy

                metrics = run_external_strategy(test_df, external_strategy)
            else:
                model = fit_model(train_df, cfg)
                metrics = backtest_on_df(
                    test_df,
                    model,
                    cfg,
                    slippage_model=slippage_model,
                    latency_ms=latency_ms,
                )
            metrics["period_start"] = start_iso
            metrics["period_end"] = end_iso
            logger.info(
                "%s to %s - Sharpe %.4f, MaxDD %.2f%%",
                metrics["period_start"],
                metrics["period_end"],
                metrics["sharpe"],
                metrics["max_drawdown"],
            )
            period_metrics.append(metrics)

        start += pd.DateOffset(months=step)

    if parallel:
        for metrics in ray.get(futures):
            logger.info(
                "%s to %s - Sharpe %.4f, MaxDD %.2f%%",
                metrics["period_start"],
                metrics["period_end"],
                metrics["sharpe"],
                metrics["max_drawdown"],
            )
            period_metrics.append(metrics)
        ray_shutdown()

    if not period_metrics:
        logger.info("No valid windows for rolling backtest")
        return {}

    avg_sharpe = float(np.mean([m["sharpe"] for m in period_metrics]))
    worst_dd = float(np.min([m["max_drawdown"] for m in period_metrics]))
    logger.info(
        "Overall average Sharpe %.4f | Worst drawdown %.2f%%",
        avg_sharpe,
        worst_dd,
    )

    return {
        "periods": period_metrics,
        "avg_sharpe": avg_sharpe,
        "worst_drawdown": worst_dd,
    }


@log_exceptions
def main():
    init_logging()
    import argparse

    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument(
        "--external-strategy",
        dest="external_strategy",
        help="Path to Freqtrade or Backtrader strategy",
    )
    parser.add_argument(
        "--evo-search",
        action="store_true",
        help="Run evolutionary multi-objective parameter search",
    )
    parser.add_argument(
        "--sim-env",
        action="store_true",
        help="Use simulated market data instead of history",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.sim_env:
        cfg["sim_env"] = True
    if args.evo_search:
        from copy import deepcopy
        from tuning.evolutionary_search import run_evolutionary_search

        def eval_fn(params: dict) -> tuple[float, float, float]:
            trial_cfg = deepcopy(cfg)
            trial_cfg.update(params)
            metrics = run_backtest(trial_cfg, external_strategy=args.external_strategy)
            total_return = metrics.get("total_return")
            return (
                -float(
                    total_return
                    if total_return is not None
                    else metrics.get("return", 0.0)
                ),
                float(metrics.get("max_drawdown", 0.0)),
                -float(metrics.get("trade_count", metrics.get("trades", 0.0))),
            )

        space = {
            "stop_loss": (0.005, 0.05, "float"),
            "take_profit": (0.005, 0.1, "float"),
        }
        run_evolutionary_search(eval_fn, space)
        return

    metrics = run_backtest(cfg, external_strategy=args.external_strategy)
    logger.info("Single period backtest:")
    for k, v in metrics.items():
        if k in {"max_drawdown", "win_rate"}:
            logger.info("%s: %.2f%%", k, v)
        else:
            logger.info("%s: %.4f", k, v)

    logger.info("Rolling backtest:")
    run_rolling_backtest(cfg, external_strategy=args.external_strategy)

    # Refresh baseline strategy scores so new baskets or strategies receive
    # historical weights immediately.
    try:
        StrategyEvaluator().run()
    except Exception:
        pass


if __name__ == "__main__":
    main()
