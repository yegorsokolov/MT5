import copy
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import ttest_ind
from skopt import gp_minimize
from skopt.space import Real, Integer
import mlflow

from train import main as train_model
from utils import load_config, update_config
from backtest import run_backtest, run_rolling_backtest
from log_utils import setup_logging, log_exceptions

logger = setup_logging()


_LOG_PATH = Path(__file__).resolve().parent / "logs" / "hyperopt_history.csv"
_LOG_PATH.parent.mkdir(exist_ok=True)


def objective(params: List[float], base_cfg: dict, results: List[dict]):
    th, stop, rsi, risk_pen, tx_cost, window = params
    test_cfg = copy.deepcopy(base_cfg)
    test_cfg["threshold"] = th
    test_cfg["trailing_stop_pips"] = int(stop)
    test_cfg["rsi_buy"] = int(rsi)
    test_cfg["rl_risk_penalty"] = float(risk_pen)
    test_cfg["rl_transaction_cost"] = float(tx_cost)
    test_cfg["backtest_window_months"] = int(window)

    metrics = run_rolling_backtest(test_cfg)
    results.append({
        "threshold": th,
        "trailing_stop_pips": int(stop),
        "rsi_buy": int(rsi),
        "rl_risk_penalty": risk_pen,
        "rl_transaction_cost": tx_cost,
        "window": int(window),
        "avg_sharpe": metrics.get("avg_sharpe", float("nan")),
    })
    return -metrics.get("avg_sharpe", -1e9)


@log_exceptions
def main():
    mlflow.set_experiment("auto_optimize")
    with mlflow.start_run():
        train_model()

        cfg = load_config()
        base_metrics, base_returns = run_backtest(cfg, return_returns=True)
        base_cv = run_rolling_backtest(cfg)

        space = [
        Real(0.5, 0.7, name="threshold"),
        Integer(10, 30, name="trailing_stop_pips"),
        Integer(50, 70, name="rsi_buy"),
        Real(0.05, 0.2, name="rl_risk_penalty"),
        Real(5e-5, 3e-4, name="rl_transaction_cost"),
        Integer(3, 12, name="backtest_window_months"),
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

        best_idx = df["avg_sharpe"].idxmax()
        best_params = df.loc[best_idx]
        best_cfg = copy.deepcopy(cfg)
        best_cfg["threshold"] = float(best_params["threshold"])
        best_cfg["trailing_stop_pips"] = int(best_params["trailing_stop_pips"])
        best_cfg["rsi_buy"] = int(best_params["rsi_buy"])
        best_cfg["rl_risk_penalty"] = float(best_params["rl_risk_penalty"])
        best_cfg["rl_transaction_cost"] = float(best_params["rl_transaction_cost"])
        best_cfg["backtest_window_months"] = int(best_params["backtest_window_months"])

        best_metrics, best_returns = run_backtest(best_cfg, return_returns=True)
        best_cv = run_rolling_backtest(best_cfg)

        stat = ttest_ind(best_returns, base_returns, equal_var=False)
        improved = (
            stat.pvalue < 0.05
            and best_metrics["sharpe"] > base_metrics["sharpe"]
            and best_cv.get("avg_sharpe", -1e9) > base_cv.get("avg_sharpe", -1e9)
        )
        reason = (
            f"bayesian optimisation avg sharpe {best_cv.get('avg_sharpe', float('nan')):.4f} p {stat.pvalue:.4f}"
        )

        if improved:
            if best_cfg["threshold"] != cfg.get("threshold"):
                update_config("threshold", best_cfg["threshold"], reason)
            if best_cfg["trailing_stop_pips"] != cfg.get("trailing_stop_pips"):
                update_config("trailing_stop_pips", best_cfg["trailing_stop_pips"], reason)
            if best_cfg["rsi_buy"] != cfg.get("rsi_buy"):
                update_config("rsi_buy", best_cfg["rsi_buy"], reason)
            if best_cfg["rl_risk_penalty"] != cfg.get("rl_risk_penalty"):
                update_config("rl_risk_penalty", best_cfg["rl_risk_penalty"], reason)
            if best_cfg["rl_transaction_cost"] != cfg.get("rl_transaction_cost"):
                update_config("rl_transaction_cost", best_cfg["rl_transaction_cost"], reason)
            if best_cfg["backtest_window_months"] != cfg.get("backtest_window_months"):
                update_config("backtest_window_months", best_cfg["backtest_window_months"], reason)

        mlflow.log_params(
            {
                "threshold": best_cfg["threshold"],
                "trailing_stop_pips": best_cfg["trailing_stop_pips"],
                "rsi_buy": best_cfg["rsi_buy"],
                "rl_risk_penalty": best_cfg["rl_risk_penalty"],
                "rl_transaction_cost": best_cfg["rl_transaction_cost"],
                "backtest_window_months": best_cfg["backtest_window_months"],
            }
        )
        mlflow.log_metrics(
            {
                "base_sharpe": base_metrics["sharpe"],
                "best_sharpe": best_metrics["sharpe"],
                "base_cv_sharpe": base_cv.get("avg_sharpe", float("nan")),
                "best_cv_sharpe": best_cv.get("avg_sharpe", float("nan")),
            }
        )


if __name__ == "__main__":
    main()
