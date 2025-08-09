import logging
from log_utils import setup_logging, log_exceptions

from pathlib import Path
from typing import List

import os
import numpy as np
import pandas as pd
import random
import torch
import gym
from gym import spaces
from stable_baselines3 import PPO, SAC, A2C

try:
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
except Exception:  # pragma: no cover - optional dependency
    SubprocVecEnv = DummyVecEnv = None  # type: ignore
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib import TRPO, RecurrentPPO

try:  # optional dependency - hierarchical options
    from sb3_contrib import HierarchicalPPO  # type: ignore
except Exception:  # pragma: no cover - algorithm may not be available
    HierarchicalPPO = None  # type: ignore

from plugins.rl_risk import RiskEnv

try:
    import gymnasium as gymn
except Exception:  # pragma: no cover - optional dependency
    gymn = None

import mlflow
from utils import load_config
from state_manager import save_checkpoint, load_latest_checkpoint
from utils.resource_monitor import monitor
from data.history import (
    load_history_parquet,
    save_history_parquet,
    load_history_config,
)
from data.features import make_features
from analysis.regime_detection import periodic_reclassification
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

setup_logging()
logger = logging.getLogger(__name__)

# Periodically refresh hardware capabilities
monitor.start()


class TradingEnv(gym.Env):
    """Trading environment supporting multiple symbols."""

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        max_position: float = 1.0,
        transaction_cost: float = 0.0001,
        risk_penalty: float = 0.1,
        var_window: int = 30,
        cvar_penalty: float = 0.0,
        cvar_window: int = 30,
        slippage_factor: float = 0.0,
        spread_source: str | None = None,
    ) -> None:
        super().__init__()

        if "Symbol" not in df.columns:
            raise ValueError("DataFrame must contain a 'Symbol' column")
        self.symbols = sorted(df["Symbol"].unique())
        wide = df.set_index(["Timestamp", "Symbol"])[features + ["mid"]].unstack(
            "Symbol"
        )
        wide.columns = [f"{sym}_{feat}" for feat, sym in wide.columns]
        wide = wide.dropna().reset_index(drop=True)

        self.df = wide
        self.features = features
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        self.var_window = var_window
        self.cvar_penalty = cvar_penalty
        self.cvar_window = cvar_window
        self.slippage_factor = slippage_factor
        self.spread_source = spread_source

        self.price_cols = [f"{sym}_mid" for sym in self.symbols]
        self.feature_cols = []
        for feat in features:
            self.feature_cols.extend([f"{sym}_{feat}" for sym in self.symbols])

        self.spread_cols = [f"{sym}_spread" for sym in self.symbols]

        self.n_symbols = len(self.symbols)
        self.action_space = spaces.Box(
            low=-max_position,
            high=max_position,
            shape=(self.n_symbols,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols),),
            dtype=np.float32,
        )
        self.start_equity = 1.0
        self.reset()

    def reset(self):
        self.i = 0
        self.equity = self.start_equity
        self.peak_equity = self.start_equity
        self.positions = np.zeros(self.n_symbols, dtype=np.float32)
        self.portfolio_returns: list[float] = []
        obs = self.df.loc[self.i, self.feature_cols].values.astype(np.float32)
        return obs

    def step(self, action):
        done = False

        action = np.clip(
            np.asarray(action, dtype=np.float32), -self.max_position, self.max_position
        )
        prices = self.df.loc[self.i, self.price_cols].values

        portfolio_ret = 0.0
        per_symbol_ret = np.zeros(self.n_symbols, dtype=np.float32)
        if self.i > 0:
            prev_prices = self.df.loc[self.i - 1, self.price_cols].values
            price_change = (prices - prev_prices) / prev_prices
            per_symbol_ret = self.positions * price_change
            portfolio_ret = per_symbol_ret.sum()
            self.equity *= 1 + portfolio_ret

        deltas = action - self.positions

        exec_prices = prices.copy()
        if self.spread_source == "column":
            try:
                spreads = self.df.loc[self.i, self.spread_cols].values
            except KeyError:
                spreads = np.zeros(self.n_symbols, dtype=np.float32)
            bids = prices - spreads / 2
            asks = prices + spreads / 2
            exec_prices = np.where(deltas > 0, asks, bids)

        slippage = np.zeros(self.n_symbols, dtype=np.float32)
        if self.slippage_factor > 0:
            slippage = np.abs(
                np.random.normal(scale=self.slippage_factor, size=self.n_symbols)
            )
            exec_prices = np.where(
                deltas > 0,
                exec_prices * (1 + slippage),
                exec_prices * (1 - slippage),
            )

        transaction_costs = np.abs(deltas) * self.transaction_cost
        slippage_costs = np.abs(deltas) * np.abs(exec_prices - prices) / prices
        costs = transaction_costs + slippage_costs
        cost_total = costs.sum()
        self.equity *= 1 - cost_total

        reward = portfolio_ret - cost_total
        self.positions = action

        self.i += 1
        if self.i >= len(self.df) - 1:
            done = True
        next_obs = self.df.loc[self.i, self.feature_cols].values.astype(np.float32)

        # risk penalty based on drawdown and variance
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.equity - self.peak_equity) / self.peak_equity

        self.portfolio_returns.append(portfolio_ret)
        risk = -abs(drawdown) * 0.1
        if len(self.portfolio_returns) >= self.var_window:
            var = np.var(self.portfolio_returns[-self.var_window :])
            risk -= self.risk_penalty * var

        if len(self.portfolio_returns) >= self.cvar_window and self.cvar_penalty > 0:
            window = np.array(self.portfolio_returns[-self.cvar_window :])
            var_threshold = np.percentile(window, 5)
            cvar = -window[window <= var_threshold].mean()
            reward -= self.cvar_penalty * cvar

        reward += risk

        info = {
            "portfolio_return": float(portfolio_ret),
            "per_symbol_returns": per_symbol_ret,
            "transaction_costs": costs,
            "execution_prices": exec_prices,
        }

        return next_obs, reward, done, info


class DiscreteTradingEnv(TradingEnv):
    """Discrete version for QRDQN."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import itertools

        levels = [-self.max_position, 0.0, self.max_position]
        self.discrete_actions = np.array(
            list(itertools.product(levels, repeat=self.n_symbols)), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.discrete_actions))

    def step(self, action):
        continuous = self.discrete_actions[int(action)]
        return super().step(continuous)


class HierarchicalTradingEnv(TradingEnv):
    """Environment with manager-worker actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Dict(
            {
                "manager": spaces.Discrete(3),
                "worker": spaces.Box(
                    low=0.0,
                    high=self.max_position,
                    shape=(self.n_symbols,),
                    dtype=np.float32,
                ),
            }
        )

    def step(self, action):
        if isinstance(action, dict):
            manager = int(action.get("manager", 1))
            worker = np.asarray(
                action.get("worker", np.zeros(self.n_symbols)), dtype=np.float32
            )
        else:
            manager, worker = action
            worker = np.asarray(worker, dtype=np.float32)
        direction = {0: -1.0, 1: 0.0, 2: 1.0}.get(manager, 0.0)
        continuous = direction * np.clip(worker, 0.0, self.max_position)
        return super().step(continuous)


class RLLibTradingEnv(TradingEnv):
    """Wrapper returning gymnasium-style tuples for RLlib."""

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        obs = super().reset()
        return obs, {}

    def step(self, action):  # type: ignore[override]
        obs, reward, done, info = super().step(action)
        return obs, reward, done, False, info


@log_exceptions
def main(
    rank: int = 0, world_size: int | None = None, cfg: dict | None = None
) -> float:
    if cfg is None:
        cfg = load_config()
    if world_size is None:
        world_size = 1
    seed = cfg.get("seed", 42) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    use_cuda = torch.cuda.is_available()
    if world_size > 1:
        backend = "nccl" if use_cuda else "gloo"
        if use_cuda:
            torch.cuda.set_device(rank)
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)
    if rank == 0:
        mlflow.set_tracking_uri(f"file:{(root / 'logs' / 'mlruns').resolve()}")
        mlflow.set_experiment("training_rl")
        mlflow.start_run()

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs = []
    for sym in symbols:
        df_sym = load_history_config(
            sym, cfg, root, validate=cfg.get("validate", False)
        )
        df_sym["Symbol"] = sym
        dfs.append(df_sym)

    df = make_features(
        pd.concat(dfs, ignore_index=True), validate=cfg.get("validate", False)
    )
    df = periodic_reclassification(df, step=cfg.get("regime_reclass_period", 500))
    features = [
        "return",
        "ma_5",
        "ma_10",
        "ma_30",
        "ma_60",
        "volatility_30",
        "spread",
        "rsi_14",
        "news_sentiment",
        "market_regime",
    ]
    features += [
        c
        for c in df.columns
        if c.startswith("cross_corr_")
        or c.startswith("factor_")
        or c.startswith("cross_mom_")
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])

    # focus training on the most recent regime
    current_regime = (
        int(df["market_regime"].iloc[-1]) if "market_regime" in df.columns else 0
    )
    df = df[df["market_regime"] == current_regime]

    size = monitor.capabilities.model_size()
    algo_cfg = cfg.get("rl_algorithm", "AUTO").upper()
    if algo_cfg == "AUTO":
        algo = "A2C" if size == "lite" else "PPO"
    else:
        algo = algo_cfg
    if algo == "PPO":
        env = TradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
            cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
            cvar_window=cfg.get("rl_cvar_window", 30),
        )
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
        )
    elif algo == "RECURRENTPPO":
        env = TradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
            cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
            cvar_window=cfg.get("rl_cvar_window", 30),
        )
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
        )
    elif algo == "A2C":
        env = TradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
            cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
            cvar_window=cfg.get("rl_cvar_window", 30),
        )
        model = A2C(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
        )
    elif algo == "A3C":
        n_envs = int(cfg.get("rl_num_envs", 4))
        n_envs = min(n_envs, os.cpu_count() or 1)

        def make_env():
            return TradingEnv(
                df,
                features,
                max_position=cfg.get("rl_max_position", 1.0),
                transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
                risk_penalty=cfg.get("rl_risk_penalty", 0.1),
                var_window=cfg.get("rl_var_window", 30),
                cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
                cvar_window=cfg.get("rl_cvar_window", 30),
            )

        if n_envs == 1:
            env = DummyVecEnv([make_env])
        else:
            env = SubprocVecEnv([make_env for _ in range(n_envs)])
        model = A2C(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
        )
    elif algo == "SAC":
        env = TradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
            cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
            cvar_window=cfg.get("rl_cvar_window", 30),
        )
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
        )
    elif algo == "TRPO":
        env = TradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
            cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
            cvar_window=cfg.get("rl_cvar_window", 30),
        )
        model = TRPO(
            "MlpPolicy",
            env,
            verbose=0,
            max_kl=cfg.get("rl_max_kl", 0.01),
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
        )
    elif algo == "HIERARCHICALPPO":
        if HierarchicalPPO is None:
            raise RuntimeError("sb3-contrib with HierarchicalPPO required")
        env = HierarchicalTradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
            cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
            cvar_window=cfg.get("rl_cvar_window", 30),
        )
        model = HierarchicalPPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
        )
    elif algo == "QRDQN":
        env = DiscreteTradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
            cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
            cvar_window=cfg.get("rl_cvar_window", 30),
        )
        model = QRDQN(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
        )
    elif algo == "RLLIB":
        if gymn is None:
            raise RuntimeError("gymnasium is required for RLlib")
        try:
            import ray
            from ray.rllib.algorithms.ppo import PPOConfig
            from ray.rllib.algorithms.ddpg import DDPGConfig
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                'RLlib not installed. Run `pip install "ray[rllib]"`'
            ) from e

        rllib_algo = cfg.get("rllib_algorithm", "PPO").upper()

        def env_creator(env_config=None):
            return RLLibTradingEnv(
                df,
                features,
                max_position=cfg.get("rl_max_position", 1.0),
                transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
                risk_penalty=cfg.get("rl_risk_penalty", 0.1),
                var_window=cfg.get("rl_var_window", 30),
                cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
                cvar_window=cfg.get("rl_cvar_window", 30),
            )

        ray.init(ignore_reinit_error=True, include_dashboard=False)
        if rllib_algo == "DDPG":
            config = (
                DDPGConfig()
                .environment(env_creator, disable_env_checking=True)
                .rollouts(num_rollout_workers=0)
                .training(
                    gamma=cfg.get("rl_gamma", 0.99),
                    lr=cfg.get("rl_learning_rate", 3e-4),
                )
                .seed(seed)
            )
        else:
            config = (
                PPOConfig()
                .environment(env_creator, disable_env_checking=True)
                .rollouts(num_rollout_workers=0)
                .training(
                    gamma=cfg.get("rl_gamma", 0.99),
                    lr=cfg.get("rl_learning_rate", 3e-4),
                )
                .seed(seed)
            )

        model = config.build()
    else:
        raise ValueError(f"Unknown rl_algorithm {algo}")

    if world_size > 1 and algo != "RLLIB":
        model.policy = DDP(model.policy, device_ids=[rank] if use_cuda else None)

    if algo == "RLLIB":
        ckpt = load_latest_checkpoint(cfg.get("checkpoint_dir"))
        start_iter = 0
        if ckpt:
            last_iter, state = ckpt
            model.restore(state["model_path"])
            start_iter = last_iter + 1
            logger.info("Resuming from checkpoint at iteration %s", last_iter)
        iters = max(1, int(cfg.get("rl_steps", 5)))
        for i in range(start_iter, iters):
            model.train()
            ckpt_path = model.save(str(root / "model_rllib"))
            save_checkpoint({"model_path": ckpt_path}, i, cfg.get("checkpoint_dir"))
        checkpoint = model.save(str(root / "model_rllib"))
        logger.info("RLlib model saved to %s", checkpoint)
        ray.shutdown()
    else:
        ckpt = load_latest_checkpoint(cfg.get("checkpoint_dir"))
        start_step = 0
        if ckpt:
            last_step, state = ckpt
            model.policy.load_state_dict(state["model"])
            model.policy.optimizer.load_state_dict(state["optimizer"])
            model.num_timesteps = last_step
            start_step = last_step
            logger.info("Resuming from checkpoint at step %s", last_step)
        total_steps = int(cfg.get("rl_steps", 5000))
        interval = int(cfg.get("checkpoint_interval", 1000))
        current = start_step
        while current < total_steps:
            learn_steps = min(interval, total_steps - current)
            model.learn(total_timesteps=learn_steps, reset_num_timesteps=False)
            current += learn_steps
            save_checkpoint(
                {
                    "model": model.policy.state_dict(),
                    "optimizer": model.policy.optimizer.state_dict(),
                    "metrics": {"timesteps": current},
                },
                current,
                cfg.get("checkpoint_dir"),
            )
        cumulative_return = 0.0
        if rank == 0:
            if algo == "RECURRENTPPO":
                rec_dir = root / "models" / "recurrent_rl"
                rec_dir.mkdir(parents=True, exist_ok=True)
                model.save(rec_dir / "recurrent_model")
                logger.info("RL model saved to %s", rec_dir / "recurrent_model.zip")
            elif algo == "HIERARCHICALPPO":
                model.save(root / "model_hierarchical")
                logger.info("RL model saved to %s", root / "model_hierarchical.zip")
            else:
                model.save(root / "model_rl")
                logger.info("RL model saved to %s", root / "model_rl.zip")
            # evaluate trained policy
            eval_size = max(2, len(df) // 5)
            eval_df = df.tail(eval_size).reset_index(drop=True)
            eval_env = TradingEnv(
                eval_df,
                features,
                max_position=cfg.get("rl_max_position", 1.0),
                transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
                risk_penalty=cfg.get("rl_risk_penalty", 0.1),
                var_window=cfg.get("rl_var_window", 30),
                cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
                cvar_window=cfg.get("rl_cvar_window", 30),
            )
            evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)
            eval_returns = np.array(eval_env.portfolio_returns)
            if eval_returns.size:
                cumulative_return = float((1 + eval_returns).prod() - 1)
                sharpe = (
                    float(np.sqrt(252) * eval_returns.mean() / eval_returns.std(ddof=0))
                    if eval_returns.std(ddof=0) > 0
                    else 0.0
                )
                equity_curve = (1 + eval_returns).cumprod()
                peak = np.maximum.accumulate(equity_curve)
                max_drawdown = float(((equity_curve - peak) / peak).min())
                mlflow.log_metric("cumulative_return", cumulative_return)
                mlflow.log_metric("sharpe_ratio", sharpe)
                mlflow.log_metric("max_drawdown", max_drawdown)

            # train risk management policy
            returns = df.sort_index()["return"].dropna()
            risk_env = RiskEnv(
                returns.values,
                lookback=cfg.get("risk_lookback_bars", 50),
                max_size=cfg.get("rl_max_position", 1.0),
            )
            risk_model = PPO(
                "MlpPolicy",
                risk_env,
                verbose=0,
                seed=seed,
                device=device,
                learning_rate=cfg.get("rl_learning_rate", 3e-4),
                gamma=cfg.get("rl_gamma", 0.99),
            )
            risk_model.learn(total_timesteps=cfg.get("rl_steps", 5000))
            models_dir = root / "models"
            models_dir.mkdir(exist_ok=True)
            risk_model.save(models_dir / "rl_risk_policy")
            logger.info("Risk policy saved to %s", models_dir / "rl_risk_policy.zip")
            mlflow.log_param("algorithm", algo)
            mlflow.log_param("steps", cfg.get("rl_steps", 5000))
            if algo == "RLLIB":
                mlflow.log_artifact(str(root / "model_rllib"))
            elif algo == "RECURRENTPPO":
                mlflow.log_artifact(str(rec_dir / "recurrent_model.zip"))
            elif algo == "HIERARCHICALPPO":
                mlflow.log_artifact(str(root / "model_hierarchical.zip"))
            else:
                mlflow.log_artifact(str(root / "model_rl.zip"))
            mlflow.log_artifact(str(models_dir / "rl_risk_policy.zip"))
            mlflow.end_run()

    if cfg.get("export"):
        from models.export import export_pytorch

        obs = eval_env.reset()[0] if eval_env is not None else env.reset()
        sample = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        export_pytorch(model.policy, sample)

    if world_size > 1:
        dist.destroy_process_group()

    return cumulative_return


def launch(cfg: dict | None = None) -> float:
    if cfg is None:
        cfg = load_config()
    use_ddp = cfg.get("ddp", monitor.capabilities.ddp())
    world_size = torch.cuda.device_count()
    if use_ddp and world_size > 1:
        mp.spawn(main, args=(world_size, cfg), nprocs=world_size)
        return 0.0
    else:
        return main(0, 1, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddp", action="store_true", help="Enable DistributedDataParallel"
    )
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search")
    parser.add_argument("--export", action="store_true", help="Export model to ONNX")
    args = parser.parse_args()
    cfg = load_config()
    if args.ddp:
        cfg["ddp"] = True
    if args.export:
        cfg["export"] = True
    if args.tune:
        from tuning.hyperopt import tune_rl

        tune_rl(cfg)
    else:
        launch(cfg)
