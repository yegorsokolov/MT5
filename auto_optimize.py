import copy
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import ttest_ind
from skopt import gp_minimize
from skopt.space import Real, Integer

from train import main as train_model
from utils import load_config, update_config
from backtest import run_backtest


_LOG_PATH = Path(__file__).resolve().parent / "logs" / "hyperopt_history.csv"
_LOG_PATH.parent.mkdir(exist_ok=True)


def objective(params: List[float], base_cfg: dict, results: List[dict]):
    th, stop, rsi = params
    test_cfg = copy.deepcopy(base_cfg)
    test_cfg["threshold"] = th
    test_cfg["trailing_stop_pips"] = int(stop)
    test_cfg["rsi_buy"] = int(rsi)

    metrics = run_backtest(test_cfg)
    results.append({
        "threshold": th,
        "trailing_stop_pips": int(stop),
        "rsi_buy": int(rsi),
        "sharpe": metrics.get("sharpe", float("nan")),
    })
    return -metrics.get("sharpe", -1e9)


def main():
    train_model()

    cfg = load_config()
    base_metrics, base_returns = run_backtest(cfg, return_returns=True)

    space = [
        Real(0.5, 0.7, name="threshold"),
        Integer(10, 30, name="trailing_stop_pips"),
        Integer(50, 70, name="rsi_buy"),
    ]

    results: List[dict] = []
    def obj(params):
        return objective(params, cfg, results)

    gp_minimize(obj, space, n_calls=15, random_state=0)

    df = pd.DataFrame(results)
    if not _LOG_PATH.exists():
        df.to_csv(_LOG_PATH, index=False)
    else:
        df.to_csv(_LOG_PATH, mode="a", header=False, index=False)

    best_idx = df["sharpe"].idxmax()
    best_params = df.loc[best_idx]
    best_cfg = copy.deepcopy(cfg)
    best_cfg["threshold"] = float(best_params["threshold"])
    best_cfg["trailing_stop_pips"] = int(best_params["trailing_stop_pips"])
    best_cfg["rsi_buy"] = int(best_params["rsi_buy"])

    best_metrics, best_returns = run_backtest(best_cfg, return_returns=True)

    stat = ttest_ind(best_returns, base_returns, equal_var=False)
    improved = (
        stat.pvalue < 0.05 and best_metrics["sharpe"] > base_metrics["sharpe"]
    )
    reason = (
        f"bayesian optimisation sharpe {best_metrics['sharpe']:.4f} p {stat.pvalue:.4f}"
    )

    if improved:
        if best_cfg["threshold"] != cfg.get("threshold"):
            update_config("threshold", best_cfg["threshold"], reason)
        if best_cfg["trailing_stop_pips"] != cfg.get("trailing_stop_pips"):
            update_config("trailing_stop_pips", best_cfg["trailing_stop_pips"], reason)
        if best_cfg["rsi_buy"] != cfg.get("rsi_buy"):
            update_config("rsi_buy", best_cfg["rsi_buy"], reason)


if __name__ == "__main__":
    main()
