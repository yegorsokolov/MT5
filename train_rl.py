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
import pickle
import json

try:  # optional dependency
    from utils.lr_scheduler import LookaheadAdamW
except Exception:  # pragma: no cover - utils may be stubbed
    LookaheadAdamW = object  # type: ignore
try:
    import torch.nn as nn
except Exception:  # pragma: no cover - torch may be stubbed
    from types import SimpleNamespace

    nn = SimpleNamespace(Module=object, Linear=lambda *a, **k: None)  # type: ignore
import gym
from gym import spaces
from datetime import datetime
try:  # optional dependency
    from stable_baselines3 import PPO, SAC, A2C
except Exception:  # pragma: no cover - optional dependency
    PPO = SAC = A2C = object  # type: ignore

try:
    from stable_baselines3.common.policies import ActorCriticPolicy
except Exception:  # pragma: no cover - optional dependency
    ActorCriticPolicy = object  # type: ignore

try:
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
except Exception:  # pragma: no cover - optional dependency
    SubprocVecEnv = DummyVecEnv = None  # type: ignore
from stable_baselines3.common.evaluation import evaluate_policy
try:  # optional dependency
    from sb3_contrib.qrdqn import QRDQN
    from sb3_contrib import TRPO, RecurrentPPO
except Exception:  # pragma: no cover - optional dependency
    QRDQN = TRPO = RecurrentPPO = object  # type: ignore

try:  # optional dependency
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch not available
    DataLoader = TensorDataset = object  # type: ignore

try:  # optional dependency - hierarchical options
    from sb3_contrib import HierarchicalPPO  # type: ignore
except Exception:  # pragma: no cover - algorithm may not be available
    HierarchicalPPO = None  # type: ignore

from plugins.rl_risk import RiskEnv
from rl.multi_objective import weighted_sum
from rl.trading_env import (
    TradingEnv,
    DiscreteTradingEnv,
    HierarchicalTradingEnv,
    RLLibTradingEnv,
)
from rl.risk_shaped_env import RiskShapedTradingEnv
from rl.macro_reward_wrapper import MacroRewardWrapper
from rl.multi_agent_env import MultiAgentTradingEnv
from rl.constrained_agent import ConstrainedAgent
from rl.meta_controller import MetaControllerDataset, train_meta_controller

try:
    import gymnasium as gymn
except Exception:  # pragma: no cover - optional dependency
    gymn = None

try:
    from rl.per import PrioritizedReplayBuffer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PrioritizedReplayBuffer = None  # type: ignore

try:
    from analytics import mlflow_client as mlflow
except Exception:  # pragma: no cover - analytics optional in tests
    try:  # fall back to direct mlflow import if available
        import mlflow  # type: ignore
    except Exception:  # pragma: no cover - mlflow missing
        from types import SimpleNamespace

        mlflow = SimpleNamespace(  # type: ignore
            set_tracking_uri=lambda *a, **k: None,
            set_experiment=lambda *a, **k: None,
            start_run=lambda *a, **k: None,
            log_param=lambda *a, **k: None,
            log_artifact=lambda *a, **k: None,
            end_run=lambda *a, **k: None,
            log_metric=lambda *a, **k: None,
            __spec__=SimpleNamespace(),
        )
from utils import load_config
from state_manager import save_checkpoint, load_latest_checkpoint
from analysis.grad_monitor import GradientMonitor, GradMonitorCallback
from analysis.market_simulator import AdversarialMarketSimulator

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

# Simulation environment for offline training
try:
    from simulation.agent_market import AgentMarketSimulator
except Exception:  # pragma: no cover - simulator optional
    AgentMarketSimulator = None  # type: ignore

try:
    from models import model_store  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    model_store = None  # type: ignore
from features import make_features
from model_registry import register_policy, save_model, get_policy_path
import joblib
from datetime import datetime
from analytics.metrics_store import record_metric, TS_PATH
from training.curriculum import CurriculumScheduler

try:
    from analysis import model_card
except Exception:  # pragma: no cover - optional dependency
    model_card = SimpleNamespace(log_model_card=lambda *a, **k: None)

try:  # optional dependency
    from models.distillation import distill_teacher_student
except Exception:  # pragma: no cover

    def distill_teacher_student(*a, **k):  # type: ignore
        return None


try:  # optional dependency
    from models.build_model import build_model, compute_scale_factor
except Exception:  # pragma: no cover

    def build_model(*a, **k):  # type: ignore
        return object()

    def compute_scale_factor() -> float:  # type: ignore
        return 1.0


try:  # optional dependency
    from models.quantize import apply_quantization
except Exception:  # pragma: no cover

    def apply_quantization(model, *a, **k):  # type: ignore
        return model


try:  # optional dependency
    from models.contrastive_encoder import initialize_model_with_contrastive
except Exception:  # pragma: no cover

    def initialize_model_with_contrastive(model):  # type: ignore
        return model


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
    NewsDrivenPolicy,
)

try:  # optional dependency
    from rl.graph_agent import GraphAgent
except Exception:  # pragma: no cover - torch may be stubbed
    GraphAgent = None  # type: ignore
try:  # optional executor for live trading integration
    from execution.rl_executor import RLExecutor
except Exception:  # pragma: no cover - optional dependency
    RLExecutor = SimpleNamespace(run=lambda *a, **k: None)
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

try:  # optional dependency
    from rl.world_model import WorldModel, Transition
except Exception:  # pragma: no cover - world model optional
    WorldModel = Transition = None  # type: ignore
try:  # optional dependency
    from core.orchestrator import Orchestrator
except Exception:  # pragma: no cover

    class Orchestrator:  # type: ignore
        @staticmethod
        def start() -> None:  # pragma: no cover - simple stub
            return None


TIERS = {"lite": 0, "standard": 1, "gpu": 2, "hpc": 3}


from strategy.self_review import self_review_strategy


class PositionClosePolicy(ActorCriticPolicy):
    """Policy network outputting position and close logits."""

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.n_symbols = action_space.shape[0] // 2
        last_dim = self.mlp_extractor.latent_dim_pi
        self.position_net = nn.Linear(last_dim, self.n_symbols)
        self.close_net = nn.Linear(last_dim, self.n_symbols)
        self._build = True

    def _get_action_dist_from_latent(self, latent_pi):
        pos = self.position_net(latent_pi)
        close = self.close_net(latent_pi)
        mean_actions = torch.cat([pos, close], dim=1)
        return self.action_dist.proba_distribution(mean_actions)


def offline_pretrain(
    model: object,
    store_path: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    grad_monitor: GradientMonitor | None = None,
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
        return 0.0
    policy = getattr(model, "policy", model)

    try:  # prefer torch if available
        import torch  # type: ignore

        if hasattr(policy, "parameters"):
            optimizer = LookaheadAdamW(policy.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()
            final_loss = 0.0
            for i in range(max(1, epochs)):
                for obs, actions, _, _, _ in dataset.iter_batches(batch_size):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    act_t = torch.as_tensor(actions, dtype=torch.float32)
                    pred = policy(obs_t)
                    loss = loss_fn(pred, act_t)
                    optimizer.zero_grad()
                    loss.backward()
                    if grad_monitor is not None:
                        trend, _ = grad_monitor.track(policy.parameters())
                        if trend == "explode":
                            for group in optimizer.param_groups:
                                group["lr"] *= 0.5
                        elif trend == "vanish":
                            for group in optimizer.param_groups:
                                group["lr"] *= 2.0
                    optimizer.step()
                    final_loss = float(loss)
                mlflow.log_metric("pretrain_loss", final_loss, step=i)
                lr_now = getattr(optimizer, "get_lr", lambda: lr)()
                mlflow.log_metric("pretrain_lr", lr_now, step=i)
            losses: List[float] = []
            for obs, actions, _, _, _ in dataset.iter_batches(batch_size):
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                act_t = torch.as_tensor(actions, dtype=torch.float32)
                with torch.no_grad():
                    pred = policy(obs_t)
                    losses.append(loss_fn(pred, act_t).item())
            final_loss = sum(losses) / max(len(losses), 1)
            mlflow.log_metric("pretrain_final_loss", final_loss)
            try:  # persist metrics locally if metrics store available
                record_metric("pretrain_final_loss", final_loss, path=TS_PATH)
            except Exception:  # pragma: no cover - metrics store optional
                pass
            dataset.close()
            return float(final_loss)
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
    err = 0.0
    for s in dataset.samples:
        x = float(s.obs[0] if isinstance(s.obs, (list, tuple)) else s.obs)
        y = float(s.action[0] if isinstance(s.action, (list, tuple)) else s.action)
        pred = w * x + b
        err += (pred - y) ** 2
    final_loss = err / max(1, len(dataset.samples))
    try:
        mlflow.log_metric("pretrain_final_loss", final_loss)
    except Exception:
        pass
    try:
        record_metric("pretrain_final_loss", final_loss, path=TS_PATH)
    except Exception:  # pragma: no cover - metrics store optional
        pass
    dataset.close()
    return float(final_loss)


def self_optimize(model: object, env: object, cfg: dict) -> None:
    """Replay recent trades and update world model/hyper-parameters.

    The routine is deliberately lightweight – it loads the most recent
    experiences from the :mod:`event_store`, fits a new :class:`WorldModel` and
    if the average reward has deteriorated adjusts the learning rate of the
    policy.  Replay statistics and tuned parameters are stored via
    :mod:`model_store`.  All operations are wrapped in ``try`` blocks so that the
    function becomes a no-op when optional dependencies are missing.
    """

    if WorldModel is None or Transition is None:
        return

    try:
        dataset = OfflineDataset(cfg.get("event_store_path"))
    except Exception:  # pragma: no cover - dataset unavailable
        return

    window = int(cfg.get("replay_window", 100))
    samples = dataset.samples[-window:]
    dataset.close()
    if not samples:
        return

    transitions = [Transition(s.obs, s.action, s.next_obs, s.reward) for s in samples]
    try:
        wm = getattr(model, "world_model")  # type: ignore[attr-defined]
    except Exception:
        wm = None
    if wm is None:
        try:
            wm = WorldModel(env.observation_space.shape[0], env.action_space.shape[0])
        except Exception:  # pragma: no cover - gym unavailable
            return
    wm.train(transitions)
    setattr(model, "world_model", wm)

    rewards = np.array([t.reward for t in transitions], dtype=float)
    mean_r = float(rewards.mean()) if rewards.size else 0.0
    if model_store is not None:
        try:
            model_store.save_replay_stats(
                {"mean_reward": mean_r, "count": len(transitions)}
            )
        except Exception:  # pragma: no cover - optional
            pass

    threshold = cfg.get("performance_threshold")
    if threshold is not None and mean_r < float(threshold):
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for group in getattr(model.policy, "optimizer").param_groups:  # type: ignore[attr-defined]
                lr = group.get("lr", cfg.get("rl_learning_rate", 3e-4))
                new_lr = lr * float(cfg.get("drift_lr_decay", 0.5))
                group["lr"] = new_lr
                if model_store is not None:
                    try:
                        model_store.save_tuned_params({"learning_rate": new_lr})
                    except Exception:  # pragma: no cover - optional
                        pass


setup_logging()
logger = logging.getLogger(__name__)
Orchestrator.start()


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
        mlflow.start_run("training_rl", cfg)

    grad_monitor = GradientMonitor(
        explode=cfg.get("grad_explode", 1e3),
        vanish=cfg.get("grad_vanish", 1e-6),
        out_dir=root / "reports" / "gradients",
    )
    grad_callback = GradMonitorCallback(
        grad_monitor,
        check_freq=cfg.get("grad_check_freq", 100),
        decay=cfg.get("grad_lr_decay", 0.5),
        growth=cfg.get("grad_lr_growth", 2.0),
    )

    adv_sim = None
    if cfg.get("adversarial_market"):
        adv_sim = AdversarialMarketSimulator(seq_len=int(cfg.get("adv_seq_len", 32)))

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs = []
    if cfg.get("sim_env") and AgentMarketSimulator is not None:
        for sym in symbols:
            sim = AgentMarketSimulator(seed=seed, steps=int(cfg.get("sim_steps", 1000)))
            _, book = sim.run()
            df_sym = sim.to_history_df(book)
            df_sym["Symbol"] = sym
            dfs.append(df_sym)
    else:
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
    from analysis import meta_learning

    if cfg.get("meta_train"):
        X_all = df[features].values.astype(np.float32)
        y_all = (df["return"].shift(-1) > 0).astype(np.float32).values[:-1]
        tasks = meta_learning._split_by_regime(
            X_all[:-1], y_all, df["market_regime"].iloc[:-1]
        )
        state = meta_learning.meta_train_policy(
            tasks, lambda: meta_learning._LinearModel(len(features))
        )
        meta_learning.save_meta_weights(state, "rl_policy")
        return 0.0

    # focus training on the most recent regime
    current_regime = (
        int(df["market_regime"].iloc[-1]) if "market_regime" in df.columns else 0
    )
    df = df[df["market_regime"] == current_regime]
    if cfg.get("fine_tune"):
        X_reg = torch.tensor(df[features].values, dtype=torch.float32)
        y_reg = torch.tensor(
            (df["return"].shift(-1) > 0).astype(float).values[:-1], dtype=torch.float32
        )
        dataset = TensorDataset(X_reg[:-1], y_reg)
        state = meta_learning.load_meta_weights("rl_policy")
        new_state, _ = meta_learning.fine_tune_model(
            state, dataset, lambda: meta_learning._LinearModel(len(features))
        )
        meta_learning.save_meta_weights(
            new_state, "rl_policy", regime=f"regime_{current_regime}"
        )
        return 0.0

    graph_model = None
    scale_factor = compute_scale_factor()
    architecture_history = [
        {"timestamp": datetime.utcnow().isoformat(), "scale_factor": scale_factor}
    ]
    if cfg.get("graph_model"):
        graph_model = build_model(len(features), cfg, scale_factor).to(device)
        if cfg.get("use_contrastive_pretrain"):
            graph_model = initialize_model_with_contrastive(graph_model)

    size = monitor.capabilities.capability_tier()
    algo_cfg = cfg.get("rl_algorithm", "AUTO").upper()
    if cfg.get("distributional"):
        algo = "QRDQN"
    elif algo_cfg == "AUTO":
        algo = "A2C" if size == "lite" else "PPO"
    else:
        algo = algo_cfg
    cvar_env_penalty = (
        0.0 if cfg.get("risk_objective") == "cvar" else cfg.get("rl_cvar_penalty", 0.0)
    )

    def maybe_wrap(env: gym.Env) -> gym.Env:
        if cfg.get("macro_reward"):
            env = MacroRewardWrapper(env)
        if cfg.get("risk_objective") == "cvar":
            from rl.risk_cvar import CVaRRewardWrapper

            env = CVaRRewardWrapper(
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
        objectives=cfg.get("rl_objectives"),
        objective_weights=cfg.get("rl_objective_weights"),
        exit_penalty=cfg.get("rl_exit_penalty", 0.001),
    )
    if cfg.get("use_news"):
        env_kwargs["news_window"] = cfg.get("news_window", 5)
    objectives = env_kwargs.get("objectives")
    if objectives is None:
        objectives = ["pnl", "hold_cost"]
    if env_kwargs["risk_penalty"] > 0 or env_kwargs["cvar_penalty"] > 0:
        if "risk" not in objectives:
            objectives.append("risk")
    env_kwargs["objectives"] = objectives
    env_cls = RiskShapedTradingEnv if cfg.get("risk_shaped") else TradingEnv

    def _policy_kwargs(scale: float) -> dict:
        if monitor.capabilities.capability_tier() == "lite":
            arch = dict(pi=[32], vf=[32])
        else:
            width = max(4, int(64 * scale))
            arch = dict(pi=[width, width], vf=[width, width])
        return {"net_arch": arch}

    if algo == "PPO":
        env = env_cls(**env_kwargs)
        env = maybe_wrap(env)
        algo_class = PPO
        policy_type = PositionClosePolicy
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
        env = env_cls(**env_kwargs)
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
        env = env_cls(**env_kwargs)
        env = maybe_wrap(env)
        algo_class = A2C
        policy_type = PositionClosePolicy
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
            env_i = env_cls(**env_kwargs)
            return maybe_wrap(env_i)

        if n_envs == 1:
            env = DummyVecEnv([make_env])
        else:
            env = SubprocVecEnv([make_env for _ in range(n_envs)])
        algo_class = A2C
        policy_type = PositionClosePolicy
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
        env = env_cls(**env_kwargs)
        env = maybe_wrap(env)
        per_kwargs = {}
        if (
            TIERS.get(monitor.capabilities.capability_tier(), 0) >= TIERS["gpu"]
            and PrioritizedReplayBuffer
        ):
            per_kwargs = {
                "replay_buffer_class": PrioritizedReplayBuffer,
                "replay_buffer_kwargs": {
                    "capacity": int(cfg.get("rl_buffer_size", 100000))
                },
            }
        algo_class = SAC
        policy_type = PositionClosePolicy
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
        env = env_cls(**env_kwargs)
        env = maybe_wrap(env)
        algo_class = TRPO
        policy_type = PositionClosePolicy
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
        if (
            TIERS.get(monitor.capabilities.capability_tier(), 0) >= TIERS["gpu"]
            and PrioritizedReplayBuffer
        ):
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

        if cfg.get("multi_agent"):

            def env_creator(env_config=None):
                env_i = MultiAgentTradingEnv(**env_kwargs)
                return maybe_wrap(env_i)

            temp_env = env_creator()
            policies = {
                sym: (
                    None,
                    temp_env.observation_space[sym],
                    temp_env.action_space[sym],
                    {},
                )
                for sym in temp_env.agents
            }
            ray.init(ignore_reinit_error=True, include_dashboard=False)
            config = (
                PPOConfig()
                .environment(env_creator, disable_env_checking=True)
                .rollouts(num_rollout_workers=0)
                .training(
                    gamma=cfg.get("rl_gamma", 0.99),
                    lr=cfg.get("rl_learning_rate", 3e-4),
                    model={"vf_share_layers": True},
                    replay_buffer_config={"type": "MultiAgentReplayBuffer"},
                )
                .multi_agent(
                    policies=policies,
                    policy_mapping_fn=lambda agent_id, *a, **k: agent_id,
                )
                .seed(seed)
            )
        else:

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

    if (
        algo != "RLLIB"
        and hasattr(model, "policy")
        and hasattr(model.policy, "optimizer")
    ):
        model.policy.optimizer = LookaheadAdamW(
            model.policy.parameters(), lr=cfg.get("rl_learning_rate", 3e-4)
        )

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
                            {
                                "timestamp": datetime.utcnow().isoformat(),
                                "scale_factor": scale_factor,
                            }
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
            Path(checkpoint),
            cfg,
            {},
            architecture_history=architecture_history,
            features=features,
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
        elif cfg.get("inverse_reward"):
            try:
                from rl.inverse_reward import learn_inverse_reward, pretrain_with_reward

                dataset = OfflineDataset(cfg.get("event_store_path"))
                params, reward_fn = learn_inverse_reward(dataset)
                pretrain_with_reward(model, dataset, reward_fn)
                if model_store is not None:
                    model_store.save_model(params, cfg, {"type": "inverse_reward"})
                logger.info(
                    "Completed inverse reward pretraining using recorded experiences"
                )
            except Exception:
                logger.exception("Inverse reward pretraining failed")
        elif cfg.get("offline_pretrain"):
            try:
                offline_pretrain(
                    model,
                    cfg.get("event_store_path"),
                    grad_monitor=grad_monitor,
                )
                logger.info("Completed offline pretraining using recorded experiences")
            except Exception:
                logger.exception("Offline pretraining failed")
        if cfg.get("world_model"):
            try:
                from rl.world_model import WorldModel, WorldModelEnv, Transition

                dataset = OfflineDataset(cfg.get("event_store_path"))
                transitions = [
                    Transition(s.obs, s.action, s.next_obs, s.reward)
                    for s in dataset.samples
                ]
                if transitions:
                    wm = WorldModel(
                        env.observation_space.shape[0], env.action_space.shape[0]
                    )
                    wm.train(transitions)
                    sim_env = WorldModelEnv(wm, env.observation_space, env.action_space)
                    if hasattr(model, "set_env"):
                        model.set_env(sim_env)
                    try:
                        model.learn(
                            total_timesteps=int(cfg.get("world_model_steps", 1000)),
                            reset_num_timesteps=False,
                            callback=grad_callback,
                        )
                    except TypeError:  # pragma: no cover - stub algos
                        model.learn(
                            total_timesteps=int(cfg.get("world_model_steps", 1000)),
                            callback=grad_callback,
                        )
                    if hasattr(model, "set_env"):
                        model.set_env(env)
                dataset.close()
                logger.info("Completed world model pretraining")
            except Exception:  # pragma: no cover - optional path
                logger.exception("World model pretraining failed")
        total_steps = int(cfg.get("rl_steps", 5000))
        interval = int(cfg.get("checkpoint_interval", 1000))
        current = start_step
        while current < total_steps:
            learn_steps = min(interval, total_steps - current)
            try:
                model.learn(
                    total_timesteps=learn_steps,
                    reset_num_timesteps=False,
                    callback=grad_callback,
                )
            except TypeError:  # pragma: no cover - stub algos may not support kwarg
                model.learn(total_timesteps=learn_steps, callback=grad_callback)

            if adv_sim is not None and hasattr(env, "price_history") and hasattr(model, "policy"):
                try:
                    history = np.array(env.price_history[-adv_sim.seq_len:])  # type: ignore
                    env.price_history[-adv_sim.seq_len:] = adv_sim.perturb(history, model.policy)  # type: ignore
                except Exception:  # pragma: no cover - environment may not support adversary
                    logger.exception("Adversarial simulator step failed")
            current += learn_steps
            if rank == 0 and hasattr(model.policy, "optimizer"):
                mlflow.log_metric("lr", model.policy.optimizer.get_lr(), step=current)
            save_checkpoint(
                {
                    "model": model.policy.state_dict(),
                    "optimizer": model.policy.optimizer.state_dict(),
                    "metrics": {"timesteps": current},
                },
                current,
                cfg.get("checkpoint_dir"),
            )
            if cfg.get("world_model"):
                try:
                    self_optimize(model, env, cfg)
                except Exception:  # pragma: no cover - optional path
                    logger.exception("Self optimisation failed")
            if rank == 0 and federated_client is not None:
                federated_client.push_update()
        cumulative_return = 0.0
        if rank == 0:
            if algo == "RECURRENTPPO":
                rec_dir = root / "models" / "recurrent_rl"
                rec_dir.mkdir(parents=True, exist_ok=True)
                policy_path = rec_dir / "recurrent_model"
                model.save(policy_path)
                logger.info("RL model saved to %s", policy_path.with_suffix(".zip"))
            elif algo == "HIERARCHICALPPO":
                policy_path = root / "model_hierarchical"
                model.save(policy_path)
                logger.info("RL model saved to %s", policy_path.with_suffix(".zip"))
            else:
                policy_path = root / "model_rl"
                model.save(policy_path)
                logger.info("RL model saved to %s", policy_path.with_suffix(".zip"))
            register_policy(
                "rl_small",
                policy_path,
                {"algo": algo, "timestamp": datetime.utcnow().isoformat()},
            )
            # evaluate trained policy
            eval_size = max(2, len(df) // 5)
            eval_df = df.tail(eval_size).reset_index(drop=True)
            eval_env = env_cls(
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
                news_window=env_kwargs.get("news_window", 0),
            )
            evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)
            eval_returns = np.array(eval_env.portfolio_returns)
            cumulative_return = 0.0
            sharpe = 0.0
            max_drawdown = 0.0
            var = 0.0
            cvar = 0.0
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
                var = float(var_threshold)
                mlflow.log_metric("value_at_risk", var)
                if np.any(eval_returns <= var_threshold):
                    cvar = -float(eval_returns[eval_returns <= var_threshold].mean())
                else:
                    cvar = 0.0
            mlflow.log_metric("cvar", cvar)
            if cfg.get("use_news"):
                record_metric("news_reward", cumulative_return, path=TS_PATH)
                record_metric("news_sharpe", sharpe, path=TS_PATH)
            else:
                record_metric("baseline_reward", cumulative_return, path=TS_PATH)
                record_metric("baseline_sharpe", sharpe, path=TS_PATH)

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
            risk_model.learn(
                total_timesteps=cfg.get("rl_steps", 5000), callback=grad_callback
            )
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
                features=features,
            )
            logger.info(
                "Distilled RL policy saved to %s", root / "model_rl_distilled.zip"
            )
            version_id = model_store.save_model(
                artifact,
                cfg,
                {"cumulative_return": cumulative_return},
                architecture_history=architecture_history,
                features=features,
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
                    features=features,
                )
                logger.info("Quantized RL policy saved to %s", q_artifact)
            model_card.generate(
                cfg,
                [root / "data" / f"{s}_history.parquet" for s in symbols],
                features,
                {
                    "cumulative_return": cumulative_return,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_drawdown,
                    "cvar": cvar,
                    "value_at_risk": var,
                },
                root / "reports" / "model_cards",
            )
            mlflow.end_run()

    # Optionally train a meta-controller from logged returns
    if cfg.get("meta_controller") and rank == 0:
        try:
            sub_returns = getattr(env, "subpolicy_returns", None)
            states = getattr(env, "state_embeddings", None)
            if sub_returns is not None and states is not None:
                ret_matrix = np.asarray(sub_returns, dtype=float)
                state_matrix = np.asarray(states, dtype=float)
            else:
                # Fallback to single-agent portfolio returns and simple regime feature
                returns_arr = np.array(
                    getattr(env, "portfolio_returns", []), dtype=float
                )
                regimes_arr = np.array(
                    getattr(env, "regimes", np.zeros_like(returns_arr)), dtype=float
                )
                if returns_arr.size == 0:
                    raise ValueError("No returns logged for meta-controller training")
                ret_matrix = np.column_stack([returns_arr, -returns_arr])
                state_matrix = regimes_arr.reshape(-1, 1)
            dataset = MetaControllerDataset(ret_matrix, state_matrix)
            train_meta_controller(dataset, epochs=int(cfg.get("meta_epochs", 10)))
            logger.info("Meta-controller trained on %d samples", len(dataset.returns))
        except Exception:  # pragma: no cover - best effort
            logger.exception("Meta-controller training failed")

    if cfg.get("export"):
        from models.export import export_pytorch

        obs = eval_env.reset()[0] if eval_env is not None else env.reset()
        sample = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        export_pytorch(model.policy, sample)

    if world_size > 1:
        dist.destroy_process_group()

    grad_monitor.plot("rl")
    return cumulative_return


def launch(cfg: dict | None = None) -> float:
    if cfg is None:
        cfg = load_config()
    curriculum_cfg = cfg.get("curriculum") if isinstance(cfg, dict) else None
    if curriculum_cfg:
        def _build_fn(stage_cfg: dict) -> callable:
            def _run_stage() -> float:
                cfg_stage = dict(cfg)
                cfg_stage.update(stage_cfg.get("config", {}))
                return main(0, 1, cfg_stage)
            return _run_stage
        scheduler = CurriculumScheduler.from_config(curriculum_cfg, _build_fn)
        if scheduler is not None:
            scheduler.run()
            return float(scheduler.final_metric)
    if cfg.get("constrained"):
        return train_constrained(cfg)
    if cfg.get("rl_exec"):
        return train_execution(cfg)
    if cfg.get("graph_rl"):
        return train_graph_rl(cfg)
    if cfg.get("hierarchical_eval"):
        return eval_hierarchical(cfg)
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


def train_constrained(cfg: dict) -> float:
    """Train :class:`ConstrainedAgent` on a tiny synthetic environment.

    The routine demonstrates how risk budgets can be enforced using a primal–
    dual update.  Rewards and costs are generated from a two-action toy
    environment where one action is riskier but higher reward.  The agent learns
    to respect ``cfg['risk_budget']`` by adjusting its dual variable.
    """

    budget = cfg.get("risk_budget", 0.0)
    risk_mgr = cfg.get("risk_manager")
    agent = ConstrainedAgent(n_actions=2, risk_budget=budget, risk_manager=risk_mgr)
    for _ in range(cfg.get("constrained_steps", 100)):
        action = agent.act(None)
        reward, cost = (1.0, 0.1) if action == 0 else (2.0, 0.5)
        agent.update(action, reward, cost)
    return agent.lambda_


def train_graph_rl(cfg: dict) -> float:
    """Train :class:`GraphAgent` using precomputed graphs.

    The implementation is intentionally lightweight and operates on synthetic
    data so it can run in unit tests without heavy dependencies.  When
    ``graph_rl`` is enabled, precomputed adjacency matrices are loaded from the
    path specified in ``cfg['graph_path']`` and a small policy network is
    trained for a few iterations.
    """

    if GraphAgent is None:  # pragma: no cover - torch dependency missing
        raise RuntimeError("GraphAgent not available")

    path = Path(cfg.get("graph_path", "data/graphs/rolling.pkl"))
    matrices = {}
    if path.exists():
        try:
            with open(path, "rb") as f:
                matrices = pickle.load(f)
        except Exception:  # pragma: no cover - loading failed
            matrices = {}

    # Create a tiny environment: rewards are +1 for selecting the last node
    num_nodes = next(iter(matrices.values())).shape[0] if matrices else 2
    features = np.eye(num_nodes)
    agent = GraphAgent(in_features=num_nodes, hidden_dim=16, num_actions=num_nodes)
    adj = next(iter(matrices.values())) if matrices else np.ones((num_nodes, num_nodes))
    for _ in range(cfg.get("graph_rl_steps", 10)):
        action = agent.act(features, adj)
        reward = 1.0 if action == num_nodes - 1 else 0.0
        agent.store(features, adj, action, reward)
        agent.train()
    return 0.0


def train_hierarchical(cfg: dict) -> float:
    """Run a lightweight training loop for :class:`HierarchicalAgent`."""

    symbol = cfg.get("symbol", "A")
    root = Path(__file__).resolve().parent
    if cfg.get("history_path"):
        df = pd.read_csv(cfg["history_path"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    else:
        df = load_history_config(symbol, cfg, root)
    df["Symbol"] = symbol
    df = make_features(df, validate=cfg.get("validate", False))

    env = HierarchicalTradingEnv(df, ["return"], max_position=1.0)
    workers = {
        "mean_reversion": MeanReversionPolicy(),
        "news": NewsDrivenPolicy(),
        "trend": TrendPolicy(),
    }
    manager = EpsilonGreedyManager(list(workers.keys()), epsilon=0.0)
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

    if cfg.get("checkpoint_dir"):
        base = Path(cfg["checkpoint_dir"])
        save_model("hier_manager", manager, {"role": "manager"}, base / "hier_manager.pkl")
        for name, pol in workers.items():
            save_model(
                f"hier_worker_{name}",
                pol,
                {"role": "worker", "name": name},
                base / f"hier_worker_{name}.pkl",
            )

    return 0.0


def eval_hierarchical(cfg: dict) -> float:
    """Evaluate a saved hierarchical policy by running a short backtest."""

    symbol = cfg.get("symbol", "A")
    root = Path(__file__).resolve().parent
    if cfg.get("history_path"):
        df = pd.read_csv(cfg["history_path"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    else:
        df = load_history_config(symbol, cfg, root)
    df["Symbol"] = symbol
    df = make_features(df, validate=cfg.get("validate", False))

    env = HierarchicalTradingEnv(df, ["return"], max_position=1.0)

    base = Path(cfg.get("checkpoint_dir", "."))
    manager_path = get_policy_path("hier_manager")
    if manager_path is None or not manager_path.exists():
        manager_path = base / "hier_manager.pkl"
    manager = joblib.load(manager_path)
    workers = {}
    for name in ["mean_reversion", "news", "trend"]:
        w_path = get_policy_path(f"hier_worker_{name}")
        if w_path is None or not w_path.exists():
            w_path = base / f"hier_worker_{name}.pkl"
        if w_path.exists():
            workers[name] = joblib.load(w_path)

    agent = HierarchicalAgent(manager, workers)
    obs = env.reset()
    for _ in range(cfg.get("hierarchical_steps", 10)):
        act = agent.act(obs)
        obs, _, done, _ = env.step(act)
        if done:
            break
    return 0.0


def train_execution(cfg: dict) -> float:
    """Train an execution policy using :class:`RLExecutor`."""
    source = cfg.get("order_book_source")
    if source is None:
        # generate a small synthetic order book for quick tests
        df = pd.DataFrame(
            {
                "Timestamp": pd.date_range("2020-01-01", periods=100, freq="s"),
                "BidPrice1": 100.0,
                "AskPrice1": 100.1,
                "BidVolume1": 10.0,
                "AskVolume1": 10.0,
            }
        )
        env = RLExecutor.make_env(df, side=cfg.get("side", "buy"))
    else:
        env = RLExecutor.make_env(source, side=cfg.get("side", "buy"))
    executor = RLExecutor(env=env)
    executor.train(steps=int(cfg.get("rl_steps", 1000)))
    if cfg.get("checkpoint_dir"):
        path = Path(cfg["checkpoint_dir"]) / "rl_executor"
        executor.save(path)
        register_policy(
            "rl_small",
            path,
            {"algo": "rl_executor", "timestamp": datetime.utcnow().isoformat()},
        )
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
        "--eval-hierarchical",
        action="store_true",
        help="Evaluate a saved hierarchical policy in a backtest",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        help="CSV file with historical prices for hierarchical agent",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory to save or load hierarchical policies",
    )
    parser.add_argument(
        "--distributional",
        action="store_true",
        help="Train a distributional (quantile-based) agent",
    )
    parser.add_argument(
        "--rl-exec",
        action="store_true",
        help="Train RL execution policy",
    )
    parser.add_argument(
        "--graph-model", action="store_true", help="Use GraphNet as policy backbone"
    )
    parser.add_argument(
        "--graph-rl",
        action="store_true",
        help="Enable graph-based RL agent",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default="data/graphs/rolling.pkl",
        help="Path to pickled adjacency matrices for graph RL",
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        help="Path to YAML/JSON strategy configuration",
    )
    parser.add_argument(
        "--offline-pretrain",
        action="store_true",
        help="Initialize policy using offline data before online fine-tuning",
    )
    parser.add_argument(
        "--inverse-reward",
        action="store_true",
        help="Pretrain policy using reward learned from expert demonstrations",
    )
    parser.add_argument(
        "--world-model",
        action="store_true",
        help="Use a learned world model for model-based pretraining",
    )
    parser.add_argument(
        "--risk-objective",
        choices=["cvar"],
        help="Apply specified risk objective to rewards",
    )
    parser.add_argument(
        "--risk-shaped",
        action="store_true",
        help="Train using risk shaped trading environment",
    )
    parser.add_argument(
        "--risk-budget",
        type=float,
        help="Portfolio risk threshold for constrained agents",
    )
    parser.add_argument(
        "--constrained",
        action="store_true",
        help="Enable constrained policy optimisation",
    )
    parser.add_argument(
        "--macro-reward",
        action="store_true",
        help="Enable macro-aware reward shaping",
    )
    parser.add_argument(
        "--use-news",
        action="store_true",
        help="Include rolling news scores as features",
    )
    parser.add_argument(
        "--sim-env",
        action="store_true",
        help="Use simulated market environment instead of historical data",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["return"],
        help="Objectives to optimise e.g. return risk cost",
    )
    parser.add_argument(
        "--multi-agent", action="store_true", help="Enable multi-agent training"
    )
    parser.add_argument(
        "--meta-controller",
        action="store_true",
        help="Train meta-controller from logged agent returns",
    )
    parser.add_argument(
        "--meta-train",
        action="store_true",
        help="Run meta-training to produce meta-initialised RL policy",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune from meta weights when regime shifts",
    )
    args = parser.parse_args()
    cfg = load_config()
    if args.strategy_config:
        path = Path(args.strategy_config)
        with path.open() as fh:
            if path.suffix in {".yaml", ".yml"}:
                import yaml  # type: ignore

                raw = yaml.safe_load(fh)
            else:
                import json

                raw = json.load(fh)
        from training.config import StrategyConfig
        from training.data.build_strategies import generate_strategy_examples
        from training.prompts.strategy_templates import epa_template

        strat_cfg = StrategyConfig(**raw)
        cfg["strategy_prompts"] = generate_strategy_examples(
            epa_template, 1, strat_cfg
        )
        review_dir = Path("logs/strategy_reviews")
        refined_prompts = []
        for i, prompt in enumerate(cfg["strategy_prompts"]):
            refined = self_review_strategy(
                prompt, epa_template, review_dir / f"strategy_{i}", strat_cfg
            )
            refined_prompts.append(refined)
        cfg["strategy_prompts"] = refined_prompts
    if args.ddp:
        cfg["ddp"] = True
    if args.export:
        cfg["export"] = True
    if args.quantize:
        cfg["quantize"] = True
    if args.hierarchical:
        cfg["hierarchical"] = True
    if args.eval_hierarchical:
        cfg["hierarchical_eval"] = True
    if args.distributional:
        cfg["distributional"] = True
    if args.rl_exec:
        cfg["rl_exec"] = True
    if args.graph_model:
        cfg["graph_model"] = True
    if args.graph_rl:
        cfg["graph_rl"] = True
        cfg["graph_path"] = args.graph_path
    if args.risk_objective:
        cfg["risk_objective"] = args.risk_objective
    if args.risk_shaped:
        cfg["risk_shaped"] = True
    if args.risk_budget is not None:
        cfg["risk_budget"] = args.risk_budget
    if args.constrained:
        cfg["constrained"] = True
    if args.macro_reward:
        cfg["macro_reward"] = True
    if args.offline_pretrain:
        cfg["offline_pretrain"] = True
    if args.inverse_reward:
        cfg["inverse_reward"] = True
    if args.world_model:
        cfg["world_model"] = True
    if args.objectives:
        objs = args.objectives
        if TIERS.get(monitor.capabilities.capability_tier(), 0) < TIERS["gpu"]:
            objs = objs[:1]
        cfg["rl_objectives"] = objs
    if args.multi_agent:
        cfg["multi_agent"] = True
    if args.meta_controller:
        cfg["meta_controller"] = True
    if args.meta_train:
        cfg["meta_train"] = True
    if args.fine_tune:
        cfg["fine_tune"] = True
    if args.use_news:
        cfg["use_news"] = True
    if args.sim_env:
        cfg["sim_env"] = True
    if args.history_path:
        cfg["history_path"] = args.history_path
    if args.checkpoint_dir:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.tune:
        from tuning.bayesian_search import run_search

        run_search(lambda c, t: launch(c), cfg)
    else:
        ray_init()
        try:
            launch(cfg)
        finally:
            ray_shutdown()
