from log_utils import setup_logging, log_exceptions

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import random
import torch
import gym
from gym import spaces
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
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
from data.history import (
    load_history_parquet,
    save_history_parquet,
    load_history_config,
)
from data.features import make_features

logger = setup_logging()


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
            worker = np.asarray(action.get("worker", np.zeros(self.n_symbols)), dtype=np.float32)
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
def main():
    cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file:{(root / 'logs' / 'mlruns').resolve()}")
    mlflow.set_experiment("training_rl")
    mlflow.start_run()

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs = []
    for sym in symbols:
        df_sym = load_history_config(sym, cfg, root)
        df_sym["Symbol"] = sym
        dfs.append(df_sym)

    df = make_features(pd.concat(dfs, ignore_index=True))
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

    algo = cfg.get("rl_algorithm", "PPO").upper()
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
        model = PPO("MlpPolicy", env, verbose=0, seed=seed)
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
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, seed=seed)
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
        model = A2C("MlpPolicy", env, verbose=0, seed=seed)
    elif algo == "A3C":
        n_envs = int(cfg.get("rl_num_envs", 4))
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

        env = SubprocVecEnv([make_env for _ in range(n_envs)])
        model = A2C("MlpPolicy", env, verbose=0, seed=seed)
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
        model = SAC("MlpPolicy", env, verbose=0, seed=seed)
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
        model = HierarchicalPPO("MlpPolicy", env, verbose=0, seed=seed)
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
        model = QRDQN("MlpPolicy", env, verbose=0, seed=seed)
    elif algo == "RLLIB":
        if gymn is None:
            raise RuntimeError("gymnasium is required for RLlib")
        try:
            import ray
            from ray.rllib.algorithms.ppo import PPOConfig
            from ray.rllib.algorithms.ddpg import DDPGConfig
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "RLlib not installed. Run `pip install \"ray[rllib]\"`"
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
                .seed(seed)
            )
        else:
            config = (
                PPOConfig()
                .environment(env_creator, disable_env_checking=True)
                .rollouts(num_rollout_workers=0)
                .seed(seed)
            )

        model = config.build()
    else:
        raise ValueError(f"Unknown rl_algorithm {algo}")

    if algo == "RLLIB":
        iters = max(1, int(cfg.get("rl_steps", 5)))
        for _ in range(iters):
            model.train()
        checkpoint = model.save(str(root / "model_rllib"))
        logger.info("RLlib model saved to %s", checkpoint)
        ray.shutdown()
    else:
        model.learn(total_timesteps=cfg.get("rl_steps", 5000))
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

    # train risk management policy
    returns = df.sort_index()["return"].dropna()
    risk_env = RiskEnv(
        returns.values,
        lookback=cfg.get("risk_lookback_bars", 50),
        max_size=cfg.get("rl_max_position", 1.0),
    )
    risk_model = PPO("MlpPolicy", risk_env, verbose=0, seed=seed)
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


if __name__ == "__main__":
    main()
