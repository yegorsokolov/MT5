import asyncio
import asyncio
import logging
from log_utils import setup_logging, log_exceptions

from pathlib import Path
from typing import List
from types import SimpleNamespace

import os
import numpy as np
import pandas as pd
import random
import torch
import gym
from gym import spaces
from datetime import datetime
from stable_baselines3 import PPO, SAC, A2C

try:
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
except Exception:  # pragma: no cover - optional dependency
    SubprocVecEnv = DummyVecEnv = None  # type: ignore
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib import TRPO, RecurrentPPO
from torch.utils.data import DataLoader, TensorDataset

try:  # optional dependency - hierarchical options
    from sb3_contrib import HierarchicalPPO  # type: ignore
except Exception:  # pragma: no cover - algorithm may not be available
    HierarchicalPPO = None  # type: ignore

from plugins.rl_risk import RiskEnv
from rl.multi_objective import weighted_sum

try:
    import gymnasium as gymn
except Exception:  # pragma: no cover - optional dependency
    gymn = None

try:
    from rl.per import PrioritizedReplayBuffer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PrioritizedReplayBuffer = None  # type: ignore

import mlflow
from utils import load_config
from state_manager import save_checkpoint, load_latest_checkpoint
try:
    from utils.resource_monitor import monitor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    monitor = SimpleNamespace(
        start=lambda: None,
        capability_tier="lite",
        capabilities=SimpleNamespace(capability_tier=lambda: "lite", ddp=lambda: False),
    )
from data.history import (
    load_history_parquet,
    save_history_parquet,
    load_history_config,
)
try:
    from models import model_store  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    model_store = None  # type: ignore
from data.features import make_features
from models.distillation import distill_teacher_student
from models.build_model import build_model, compute_scale_factor
from models.quantize import apply_quantization
from models.contrastive_encoder import initialize_model_with_contrastive
try:
    from analysis.regime_detection import periodic_reclassification  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    def periodic_reclassification(df, step=500):  # type: ignore
        return df
from rl.hierarchical_agent import (
    HierarchicalAgent,
    EpsilonGreedyManager,
    TrendPolicy,
    MeanReversionPolicy,
)
import argparse
try:
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
except Exception:  # pragma: no cover - optional dependency
    dist = mp = None  # type: ignore
    class DDP:  # type: ignore
        pass
from ray_utils import (
    init as ray_init,
    shutdown as ray_shutdown,
    cluster_available,
    submit,
)
from rl.offline_dataset import OfflineDataset
from event_store import EventStore
from core.orchestrator import Orchestrator

TIERS = {"lite": 0, "standard": 1, "gpu": 2, "hpc": 3}


def offline_pretrain(
    model: object,
    store_path: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    """Pretrain ``model``'s policy using offline experiences.

    The routine performs a simple behaviour cloning update over experiences
    recorded in the :mod:`event_store`.  If PyTorch is available and the policy
    exposes parameters, gradient descent is executed with an MSE loss.  In
    environments without PyTorch the function falls back to a very small
    implementation that updates ``weight`` and ``bias`` attributes of a linear
    policy using basic gradient steps.
    """

    dataset = OfflineDataset(EventStore(store_path) if store_path else EventStore())
    if len(dataset) == 0:
        dataset.close()
        return
    policy = getattr(model, "policy", model)

    try:  # prefer torch if available
        import torch  # type: ignore

        if hasattr(policy, "parameters"):
            optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()
            for _ in range(max(1, epochs)):
                for obs, actions, _, _, _ in dataset.iter_batches(batch_size):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    act_t = torch.as_tensor(actions, dtype=torch.float32)
                    pred = policy(obs_t)
                    loss = loss_fn(pred, act_t)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            dataset.close()
            return
    except Exception:  # pragma: no cover - torch not available
        pass

    # Fallback: assume simple linear policy with scalar weight and bias
    w = float(getattr(policy, "weight", 0.0))
    b = float(getattr(policy, "bias", 0.0))
    for _ in range(max(1, epochs)):
        for obs, actions, _, _, _ in dataset.iter_batches(batch_size):
            for o, a in zip(obs, actions):
                x = float(o[0] if isinstance(o, (list, tuple)) else o)
                y = float(a[0] if isinstance(a, (list, tuple)) else a)
                pred = w * x + b
                grad = pred - y
                w -= lr * 2 * grad * x
                b -= lr * 2 * grad
    policy.weight = w
    policy.bias = b
    dataset.close()

setup_logging()
logger = logging.getLogger(__name__)
Orchestrator.start()


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
        objectives: list[str] | None = None,
        objective_weights: list[float] | None = None,
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
        self.objectives = objectives or ["return"]
        if objective_weights is None:
            objective_weights = [1.0] * len(self.objectives)
        self.objective_weights = np.asarray(objective_weights, dtype=np.float32)

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

        reward_components: list[float] = []
        objective_map: dict[str, float] = {}
        if "return" in self.objectives:
            reward_components.append(portfolio_ret)
            objective_map["return"] = float(portfolio_ret)
        if "cost" in self.objectives:
            reward_components.append(-cost_total)
            objective_map["cost"] = float(-cost_total)
        reward = 0.0
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
            risk -= self.cvar_penalty * cvar

        if "risk" in self.objectives:
            reward_components.append(risk)
            objective_map["risk"] = float(risk)

        if reward_components:
            reward = weighted_sum(
                np.asarray(reward_components, dtype=np.float32),
                self.objective_weights[: len(reward_components)],
            )

        info = {
            "portfolio_return": float(portfolio_ret),
            "per_symbol_returns": per_symbol_ret,
            "transaction_costs": costs,
            "execution_prices": exec_prices,
        }
        if objective_map:
            info["objectives"] = objective_map

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
    if hasattr(torch, "device"):
        device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    else:  # pragma: no cover - torch may be stubbed
        device = "cpu"
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

    concat_df = pd.concat(dfs, ignore_index=True)
    try:
        df = make_features(concat_df, validate=cfg.get("validate", False))
    except TypeError:  # pragma: no cover - stub may not accept kwargs
        df = make_features(concat_df)
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

    graph_model = None
    scale_factor = compute_scale_factor()
    architecture_history = [
        {"timestamp": datetime.utcnow().isoformat(), "scale_factor": scale_factor}
    ]
    if cfg.get("graph_model"):
        graph_model = build_model(len(features), cfg, scale_factor).to(device)
        graph_model = initialize_model_with_contrastive(graph_model)

    size = monitor.capabilities.capability_tier()
    algo_cfg = cfg.get("rl_algorithm", "AUTO").upper()
    if algo_cfg == "AUTO":
        algo = "A2C" if size == "lite" else "PPO"
    else:
        algo = algo_cfg
    cvar_env_penalty = 0.0 if cfg.get("risk_objective") == "cvar" else cfg.get("rl_cvar_penalty", 0.0)

    def maybe_wrap(env: gym.Env) -> gym.Env:
        if cfg.get("risk_objective") == "cvar":
            from rl.risk_cvar import CVaRRewardWrapper
            return CVaRRewardWrapper(
                env,
                penalty=cfg.get("rl_cvar_penalty", 0.0),
                window=cfg.get("rl_cvar_window", 30),
            )
        return env
    env_kwargs = dict(
        df=df,
        features=features,
        max_position=cfg.get("rl_max_position", 1.0),
        transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
        risk_penalty=cfg.get("rl_risk_penalty", 0.1),
        var_window=cfg.get("rl_var_window", 30),
        cvar_penalty=cvar_env_penalty,
        cvar_window=cfg.get("rl_cvar_window", 30),
        objectives=cfg.get("rl_objectives", ["return"]),
        objective_weights=cfg.get("rl_objective_weights"),
    )

    def _policy_kwargs(scale: float) -> dict:
        width = max(4, int(64 * scale))
        return {"net_arch": [width, width]}

    if algo == "PPO":
        env = TradingEnv(**env_kwargs)
        env = maybe_wrap(env)
        algo_class = PPO
        policy_type = "MlpPolicy"
        model = algo_class(
            policy_type,
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
            policy_kwargs=_policy_kwargs(scale_factor),
        )
    elif algo == "RECURRENTPPO":
        env = TradingEnv(**env_kwargs)
        env = maybe_wrap(env)
        algo_class = RecurrentPPO
        policy_type = "MlpLstmPolicy"
        model = algo_class(
            policy_type,
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
            policy_kwargs=_policy_kwargs(scale_factor),
        )
    elif algo == "A2C":
        env = TradingEnv(**env_kwargs)
        env = maybe_wrap(env)
        algo_class = A2C
        policy_type = "MlpPolicy"
        model = algo_class(
            policy_type,
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
            policy_kwargs=_policy_kwargs(scale_factor),
        )
    elif algo == "A3C":
        n_envs = int(cfg.get("rl_num_envs", 4))
        n_envs = min(n_envs, os.cpu_count() or 1)

        def make_env():
            env_i = TradingEnv(**env_kwargs)
            return maybe_wrap(env_i)

        if n_envs == 1:
            env = DummyVecEnv([make_env])
        else:
            env = SubprocVecEnv([make_env for _ in range(n_envs)])
        algo_class = A2C
        policy_type = "MlpPolicy"
        model = algo_class(
            policy_type,
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
            policy_kwargs=_policy_kwargs(scale_factor),
        )
    elif algo == "SAC":
        env = TradingEnv(**env_kwargs)
        env = maybe_wrap(env)
        per_kwargs = {}
        if TIERS.get(monitor.capabilities.capability_tier(), 0) >= TIERS["gpu"] and PrioritizedReplayBuffer:
            per_kwargs = {
                "replay_buffer_class": PrioritizedReplayBuffer,
                "replay_buffer_kwargs": {
                    "capacity": int(cfg.get("rl_buffer_size", 100000))
                },
            }
        algo_class = SAC
        policy_type = "MlpPolicy"
        try:
            model = algo_class(
                policy_type,
                env,
                verbose=0,
                seed=seed,
                device=device,
                learning_rate=cfg.get("rl_learning_rate", 3e-4),
                gamma=cfg.get("rl_gamma", 0.99),
                policy_kwargs=_policy_kwargs(scale_factor),
                **per_kwargs,
            )
        except TypeError:  # pragma: no cover - algorithm may not support PER kwargs
            model = algo_class(
                policy_type,
                env,
                verbose=0,
                seed=seed,
                device=device,
                learning_rate=cfg.get("rl_learning_rate", 3e-4),
                gamma=cfg.get("rl_gamma", 0.99),
                policy_kwargs=_policy_kwargs(scale_factor),
            )
    elif algo == "TRPO":
        env = TradingEnv(**env_kwargs)
        env = maybe_wrap(env)
        algo_class = TRPO
        policy_type = "MlpPolicy"
        model = algo_class(
            policy_type,
            env,
            verbose=0,
            max_kl=cfg.get("rl_max_kl", 0.01),
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
            policy_kwargs=_policy_kwargs(scale_factor),
        )
    elif algo == "HIERARCHICALPPO":
        if HierarchicalPPO is None:
            raise RuntimeError("sb3-contrib with HierarchicalPPO required")
        env = HierarchicalTradingEnv(**env_kwargs)
        env = maybe_wrap(env)
        algo_class = HierarchicalPPO
        policy_type = "MlpPolicy"
        model = algo_class(
            policy_type,
            env,
            verbose=0,
            seed=seed,
            device=device,
            learning_rate=cfg.get("rl_learning_rate", 3e-4),
            gamma=cfg.get("rl_gamma", 0.99),
            policy_kwargs=_policy_kwargs(scale_factor),
        )
    elif algo == "QRDQN":
        env = DiscreteTradingEnv(**env_kwargs)
        env = maybe_wrap(env)
        per_kwargs = {}
        if TIERS.get(monitor.capabilities.capability_tier(), 0) >= TIERS["gpu"] and PrioritizedReplayBuffer:
            per_kwargs = {
                "replay_buffer_class": PrioritizedReplayBuffer,
                "replay_buffer_kwargs": {
                    "capacity": int(cfg.get("rl_buffer_size", 100000))
                },
            }
        algo_class = QRDQN
        policy_type = "MlpPolicy"
        try:
            model = algo_class(
                policy_type,
                env,
                verbose=0,
                seed=seed,
                device=device,
                learning_rate=cfg.get("rl_learning_rate", 3e-4),
                gamma=cfg.get("rl_gamma", 0.99),
                policy_kwargs=_policy_kwargs(scale_factor),
                **per_kwargs,
            )
        except TypeError:  # pragma: no cover - algorithm may not support PER kwargs
            model = algo_class(
                policy_type,
                env,
                verbose=0,
                seed=seed,
                device=device,
                learning_rate=cfg.get("rl_learning_rate", 3e-4),
                gamma=cfg.get("rl_gamma", 0.99),
                policy_kwargs=_policy_kwargs(scale_factor),
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
            env_i = RLLibTradingEnv(**env_kwargs)
            return maybe_wrap(env_i)

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

    if algo != "RLLIB":
        def _watch_model() -> None:
            async def _watch() -> None:
                q = monitor.subscribe()
                nonlocal model, scale_factor
                while True:
                    await q.get()
                    new_scale = compute_scale_factor()
                    if new_scale != scale_factor:
                        params = model.get_parameters()
                        model_new = algo_class(
                            policy_type,
                            env,
                            verbose=0,
                            seed=seed,
                            device=device,
                            learning_rate=cfg.get("rl_learning_rate", 3e-4),
                            gamma=cfg.get("rl_gamma", 0.99),
                            policy_kwargs=_policy_kwargs(new_scale),
                        )
                        model_new.set_parameters(params, exact_match=False)
                        model = model_new
                        scale_factor = new_scale
                        architecture_history.append(
                            {"timestamp": datetime.utcnow().isoformat(), "scale_factor": scale_factor}
                        )
                        logger.info(
                            "Hot-reloaded RL model with scale factor %s", scale_factor
                        )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            loop.create_task(_watch())

        _watch_model()

    if world_size > 1 and algo != "RLLIB":
        model.policy = DDP(model.policy, device_ids=[rank] if use_cuda else None)

    federated_cfg = cfg.get("federated", {})
    federated_client = None
    if federated_cfg.get("enabled") and algo != "RLLIB":
        from federated.client import FederatedClient

        target = model.policy if hasattr(model, "policy") else model
        federated_client = FederatedClient(
            federated_cfg["server_url"],
            federated_cfg["api_key"],
            target,
            cfg.get("checkpoint_dir"),
        )
        federated_client.fetch_global()

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
        version_id = model_store.save_model(
            Path(checkpoint), cfg, {}, architecture_history=architecture_history
        )
        logger.info("Registered model version %s", version_id)
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
        elif cfg.get("offline_pretrain"):
            try:
                offline_pretrain(model, cfg.get("event_store_path"))
                logger.info("Completed offline pretraining using recorded experiences")
            except Exception:
                logger.exception("Offline pretraining failed")
        total_steps = int(cfg.get("rl_steps", 5000))
        interval = int(cfg.get("checkpoint_interval", 1000))
        current = start_step
        while current < total_steps:
            learn_steps = min(interval, total_steps - current)
            try:
                model.learn(total_timesteps=learn_steps, reset_num_timesteps=False)
            except TypeError:  # pragma: no cover - stub algos may not support kwarg
                model.learn(total_timesteps=learn_steps)
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
            if rank == 0 and federated_client is not None:
                federated_client.push_update()
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
                cvar_penalty=0.0,
                cvar_window=cfg.get("rl_cvar_window", 30),
                objectives=cfg.get("rl_objectives", ["return"]),
                objective_weights=cfg.get("rl_objective_weights"),
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
                var_threshold = np.percentile(eval_returns, 5)
                if np.any(eval_returns <= var_threshold):
                    cvar = -float(eval_returns[eval_returns <= var_threshold].mean())
                else:
                    cvar = 0.0
            mlflow.log_metric("cvar", cvar)

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
            if algo == "RECURRENTPPO":
                artifact = rec_dir / "recurrent_model.zip"
            elif algo == "HIERARCHICALPPO":
                artifact = root / "model_hierarchical.zip"
            elif algo == "RLLIB":
                artifact = root / "model_rllib"
            else:
                artifact = root / "model_rl.zip"
            if TIERS.get(monitor.capabilities.capability_tier(), 0) >= TIERS["gpu"]:
                student_model = A2C(
                    "MlpPolicy",
                    env,
                    seed=seed,
                    device=device,
                    verbose=0,
                    learning_rate=cfg.get("rl_learning_rate", 3e-4),
                )
                obs_samples = []
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                for _ in range(min(256, len(df))):
                    obs_samples.append(obs)
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _ = env.step(action)
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    if done:
                        obs = env.reset()
                        if isinstance(obs, tuple):
                            obs = obs[0]
                obs_tensor = torch.tensor(np.array(obs_samples), dtype=torch.float32)
                loader = DataLoader(
                    TensorDataset(obs_tensor, torch.zeros(len(obs_tensor))),
                    batch_size=32,
                )
                distill_teacher_student(
                    model.policy,
                    student_model.policy,
                    loader,
                    epochs=cfg.get("distill_epochs", 1),
                )
                student_model.save(root / "model_rl_distilled")
                model_store.save_model(
                    root / "model_rl_distilled.zip",
                    {**cfg, "distilled_from": str(artifact)},
                    {"cumulative_return": cumulative_return},
                    architecture_history=architecture_history,
                )
                logger.info(
                    "Distilled RL policy saved to %s", root / "model_rl_distilled.zip"
                )
            version_id = model_store.save_model(
                artifact,
                cfg,
                {"cumulative_return": cumulative_return},
                architecture_history=architecture_history,
            )
            logger.info("Registered model version %s", version_id)
            if cfg.get("quantize"):
                q_policy = apply_quantization(model.policy)
                model.policy = q_policy
                q_artifact = root / "model_rl_quantized.zip"
                model.save(q_artifact)
                model_store.save_model(
                    q_artifact,
                    {**cfg, "quantized": True},
                    {"cumulative_return": cumulative_return},
                    architecture_history=architecture_history,
                )
                logger.info("Quantized RL policy saved to %s", q_artifact)
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
    if cfg.get("hierarchical"):
        return train_hierarchical(cfg)
    if cluster_available():
        seeds = cfg.get("seeds", [cfg.get("seed", 42)])
        results = []
        for s in seeds:
            cfg_s = dict(cfg)
            cfg_s["seed"] = s
            results.append(submit(main, 0, 1, cfg_s))
        return float(results[0] if results else 0.0)
    use_ddp = cfg.get("ddp", monitor.capabilities.ddp())
    world_size = torch.cuda.device_count()
    if use_ddp and world_size > 1:
        mp.spawn(main, args=(world_size, cfg), nprocs=world_size)
        return 0.0
    else:
        return main(0, 1, cfg)


def train_hierarchical(cfg: dict) -> float:
    """Run a lightweight training loop for :class:`HierarchicalAgent`."""
    # create a tiny synthetic dataset to drive the environment
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=20, freq="min"),
            "Symbol": ["A"] * 20,
            "mid": np.linspace(1.0, 1.1, 20),
            "return": np.random.randn(20) / 100,
        }
    )
    env = HierarchicalTradingEnv(df, ["return"], max_position=1.0)
    manager = EpsilonGreedyManager(["trend", "mean_reversion"], epsilon=0.0)
    workers = {"trend": TrendPolicy(), "mean_reversion": MeanReversionPolicy()}
    agent = HierarchicalAgent(manager, workers)
    obs = env.reset()
    for _ in range(cfg.get("hierarchical_steps", 10)):
        act = agent.act(obs)
        next_obs, reward, done, _ = env.step(act)
        agent.store(obs, act, reward, next_obs, done)
        agent.train(batch_size=8)
        obs = next_obs
        if done:
            obs = env.reset()
    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddp", action="store_true", help="Enable DistributedDataParallel"
    )
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search")
    parser.add_argument("--export", action="store_true", help="Export model to ONNX")
    parser.add_argument("--quantize", action="store_true", help="Save quantized model")
    parser.add_argument(
        "--hierarchical", action="store_true", help="Use hierarchical agent"
    )
    parser.add_argument(
        "--graph-model", action="store_true", help="Use GraphNet as policy backbone"
    )
    parser.add_argument(
        "--offline-pretrain",
        action="store_true",
        help="Initialize policy using offline data before online fine-tuning",
    )
    parser.add_argument(
        "--risk-objective",
        choices=["cvar"],
        help="Apply specified risk objective to rewards",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["return"],
        help="Objectives to optimise e.g. return risk cost",
    )
    args = parser.parse_args()
    cfg = load_config()
    if args.ddp:
        cfg["ddp"] = True
    if args.export:
        cfg["export"] = True
    if args.quantize:
        cfg["quantize"] = True
    if args.hierarchical:
        cfg["hierarchical"] = True
    if args.graph_model:
        cfg["graph_model"] = True
    if args.risk_objective:
        cfg["risk_objective"] = args.risk_objective
    if args.offline_pretrain:
        cfg["offline_pretrain"] = True
    if args.objectives:
        objs = args.objectives
        if TIERS.get(monitor.capabilities.capability_tier(), 0) < TIERS["gpu"]:
            objs = objs[:1]
        cfg["rl_objectives"] = objs
    if args.tune:
        from tuning.hyperopt import tune_rl

        tune_rl(cfg)
    else:
        ray_init()
        try:
            launch(cfg)
        finally:
            ray_shutdown()
