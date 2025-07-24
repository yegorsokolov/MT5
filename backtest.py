"""Simple backtesting for the Adaptive MT5 bot."""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from utils import load_config
from dataset import load_history, load_history_parquet, make_features
import ray

logger = setup_logging()
from log_utils import setup_logging, log_exceptions


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


def trailing_stop(entry_price: float, current_price: float, stop: float, distance: float) -> float:
    """Update trailing stop based on price movement."""
    if current_price - distance > stop:
        return current_price - distance
    return stop


def compute_metrics(returns: pd.Series) -> dict:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    sharpe = np.sqrt(252) * returns.mean() / returns.std(ddof=0)
    return {
        "sharpe": sharpe,
        "max_drawdown": drawdown.min() * 100,
        "total_return": cumulative.iloc[-1] - 1,
        "win_rate": (returns > 0).mean() * 100,
    }


def fit_model(train_df: pd.DataFrame, cfg: dict) -> Pipeline:
    """Train a LightGBM model on the given dataframe."""
    feats = feature_columns(train_df)
    X = train_df[feats]
    y = (train_df["return"].shift(-1) > 0).astype(int)
    steps = []
    if cfg.get("use_scaler", True):
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", LGBMClassifier(n_estimators=200, random_state=42)))
    pipe = Pipeline(steps)
    pipe.fit(X, y)
    return pipe


def backtest_on_df(
    df: pd.DataFrame, model, cfg: dict, *, return_returns: bool = False
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
    return_returns: bool, optional
        When ``True`` the list of trade returns is also returned for
        statistical testing.
    """
    feats = feature_columns(df)
    probs = model.predict_proba(df[feats])[:, 1]

    threshold = cfg.get("threshold", 0.55)
    distance = cfg.get("trailing_stop_pips", 20) * 1e-4

    in_position = False
    entry = 0.0
    stop = 0.0
    returns = []

    for price, prob in zip(df["mid"], probs):
        if not in_position and prob > threshold:
            in_position = True
            entry = price
            stop = price - distance
            continue
        if in_position:
            stop = trailing_stop(entry, price, stop, distance)
            if price <= stop:
                returns.append((price - entry) / entry)
                in_position = False

    series = pd.Series(returns)
    metrics = compute_metrics(series)
    if return_returns:
        return metrics, series
    return metrics


def run_backtest(cfg: dict, *, return_returns: bool = False) -> dict | tuple[dict, pd.Series]:
    data_path = Path(__file__).resolve().parent / "data" / "history.parquet"
    if not data_path.exists():
        raise FileNotFoundError("Historical history.parquet not found under data/")

    model = joblib.load(Path(__file__).resolve().parent / "model.joblib")

    df = load_history_parquet(data_path)
    df = make_features(df)
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes
    df = df[df.get("Symbol").isin([cfg.get("symbol")])]

    return backtest_on_df(df, model, cfg, return_returns=return_returns)


@ray.remote
def _backtest_window(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict, start: str, end: str) -> dict:
    """Train a model on ``train_df`` and backtest on ``test_df``."""
    model = fit_model(train_df, cfg)
    metrics = backtest_on_df(test_df, model, cfg)
    metrics["period_start"] = start
    metrics["period_end"] = end
    return metrics


def run_rolling_backtest(cfg: dict) -> dict:
    """Perform rolling train/test backtests and aggregate metrics."""
    data_path = Path(__file__).resolve().parent / "data" / "history.parquet"
    if not data_path.exists():
        raise FileNotFoundError("Historical history.parquet not found under data/")

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

    parallel = cfg.get("parallel_backtest", False)
    if parallel:
        ray.init(num_cpus=cfg.get("ray_num_cpus"))
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
            futures.append(_backtest_window.remote(train_df, test_df, cfg, start_iso, end_iso))
        else:
            model = fit_model(train_df, cfg)
            metrics = backtest_on_df(test_df, model, cfg)
            metrics["period_start"] = start_iso
            metrics["period_end"] = end_iso
            print(
                f"{metrics['period_start']} to {metrics['period_end']} - "
                f"Sharpe {metrics['sharpe']:.4f}, MaxDD {metrics['max_drawdown']:.2f}%"
            )
            period_metrics.append(metrics)

        start += pd.DateOffset(months=step)

    if parallel:
        for metrics in ray.get(futures):
            print(
                f"{metrics['period_start']} to {metrics['period_end']} - "
                f"Sharpe {metrics['sharpe']:.4f}, MaxDD {metrics['max_drawdown']:.2f}%"
            )
            period_metrics.append(metrics)
        ray.shutdown()

    if not period_metrics:
        print("No valid windows for rolling backtest")
        return {}

    avg_sharpe = float(np.mean([m["sharpe"] for m in period_metrics]))
    worst_dd = float(np.min([m["max_drawdown"] for m in period_metrics]))
    print(f"Overall average Sharpe {avg_sharpe:.4f} | Worst drawdown {worst_dd:.2f}%")

    return {
        "periods": period_metrics,
        "avg_sharpe": avg_sharpe,
        "worst_drawdown": worst_dd,
    }


@log_exceptions
def main():
    cfg = load_config()
    metrics = run_backtest(cfg)
    print("Single period backtest:")
    for k, v in metrics.items():
        if k in {"max_drawdown", "win_rate"}:
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.4f}")

    print("\nRolling backtest:")
    run_rolling_backtest(cfg)


if __name__ == "__main__":
    main()
