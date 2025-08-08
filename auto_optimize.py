import copy
from pathlib import Path
from typing import Dict, List

import logging
import pandas as pd
from scipy.stats import ttest_ind
import optuna
import mlflow

from train import main as train_model
from utils import load_config, update_config
from backtest import run_backtest, run_rolling_backtest
from log_utils import setup_logging, log_exceptions

setup_logging()
logger = logging.getLogger(__name__)


_LOG_PATH = Path(__file__).resolve().parent / "logs" / "optuna_history.csv"
_LOG_PATH.parent.mkdir(exist_ok=True)


@log_exceptions
def main():
    mlflow.set_experiment("auto_optimize")
    with mlflow.start_run():
        train_model()

        cfg = load_config()
        base_metrics, base_returns = run_backtest(cfg, return_returns=True)
        base_cv = run_rolling_backtest(cfg)

        results: List[Dict[str, float]] = []

        def evaluate(params: Dict[str, float]) -> float:
            test_cfg = copy.deepcopy(cfg)
            test_cfg.update(params)
            metrics = run_rolling_backtest(test_cfg)
            val = metrics.get("avg_sharpe", float("nan"))
            results.append({**params, "avg_sharpe": val})
            return val

        def sample_params(trial: optuna.trial.Trial) -> Dict[str, float]:
            return {
                "threshold": trial.suggest_float("threshold", 0.5, 0.7),
                "trailing_stop_pips": trial.suggest_int("trailing_stop_pips", 10, 30),
                "rsi_buy": trial.suggest_int("rsi_buy", 50, 70),
                "rl_max_position": trial.suggest_float("rl_max_position", 0.5, 2.0),
                "rl_risk_penalty": trial.suggest_float("rl_risk_penalty", 0.05, 0.2),
                "rl_transaction_cost": trial.suggest_float(
                    "rl_transaction_cost", 5e-5, 3e-4
                ),
                "rl_max_kl": trial.suggest_float("rl_max_kl", 0.005, 0.05),
                "backtest_window_months": trial.suggest_int(
                    "backtest_window_months", 3, 12
                ),
            }

        def objective_optuna(trial: optuna.trial.Trial) -> float:
            params = sample_params(trial)
            return evaluate(params)

        def objective_tune(config: Dict[str, float]):
            val = evaluate(config)
            tune.report(avg_sharpe=val)

        use_ray = False
        try:
            import ray  # type: ignore
            from ray import tune  # type: ignore
            from ray.tune.search.optuna import OptunaSearch  # type: ignore

            if cfg.get("use_ray"):
                ray.init(address="auto")
                use_ray = True
        except Exception:
            tune = None  # type: ignore

        if use_ray and tune is not None:
            search_alg = OptunaSearch(metric="avg_sharpe", mode="max")
            search_space = {
                "threshold": tune.uniform(0.5, 0.7),
                "trailing_stop_pips": tune.randint(10, 30),
                "rsi_buy": tune.randint(50, 70),
                "rl_max_position": tune.uniform(0.5, 2.0),
                "rl_risk_penalty": tune.uniform(0.05, 0.2),
                "rl_transaction_cost": tune.uniform(5e-5, 3e-4),
                "rl_max_kl": tune.uniform(0.005, 0.05),
                "backtest_window_months": tune.randint(3, 12),
            }
            tune.run(
                objective_tune,
                search_alg=search_alg,
                num_samples=15,
                config=search_space,
                resources_per_trial=cfg.get("ray_resources", {}),
            )
            ray.shutdown()
        else:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective_optuna, n_trials=15)

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
        best_cfg["rl_max_position"] = float(best_params["rl_max_position"])
        best_cfg["rl_risk_penalty"] = float(best_params["rl_risk_penalty"])
        best_cfg["rl_transaction_cost"] = float(best_params["rl_transaction_cost"])
        best_cfg["rl_max_kl"] = float(best_params.get("rl_max_kl", cfg.get("rl_max_kl", 0.01)))
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
            if best_cfg["rl_max_position"] != cfg.get("rl_max_position"):
                try:
                    update_config("rl_max_position", best_cfg["rl_max_position"], reason)
                except ValueError:
                    logger.warning("rl_max_position modification blocked by risk rules")
            if best_cfg["rl_transaction_cost"] != cfg.get("rl_transaction_cost"):
                update_config("rl_transaction_cost", best_cfg["rl_transaction_cost"], reason)
            if best_cfg["rl_max_kl"] != cfg.get("rl_max_kl", 0.01):
                update_config("rl_max_kl", best_cfg["rl_max_kl"], reason)
            if best_cfg["backtest_window_months"] != cfg.get("backtest_window_months"):
                update_config("backtest_window_months", best_cfg["backtest_window_months"], reason)

        mlflow.log_params(
            {
                "threshold": best_cfg["threshold"],
                "trailing_stop_pips": best_cfg["trailing_stop_pips"],
                "rsi_buy": best_cfg["rsi_buy"],
                "rl_max_position": best_cfg["rl_max_position"],
                "rl_risk_penalty": best_cfg["rl_risk_penalty"],
                "rl_transaction_cost": best_cfg["rl_transaction_cost"],
                "rl_max_kl": best_cfg["rl_max_kl"],
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
