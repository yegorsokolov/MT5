"""Reinforcement learning utilities."""

try:  # optional dependency on gym
    from .risk_cvar import CVaRRewardWrapper
except Exception:  # pragma: no cover - gym may be stubbed
    CVaRRewardWrapper = object  # type: ignore
from .world_model import WorldModel, WorldModelEnv, Transition
from .trading_env import (
    TradingEnv,
    DiscreteTradingEnv,
    HierarchicalTradingEnv,
    RLLibTradingEnv,
)
from .risk_shaped_env import RiskShapedTradingEnv
from .multi_agent_env import MultiAgentTradingEnv
from .distributional_agent import DistributionalAgent, MeanAgent

__all__ = [
    "CVaRRewardWrapper",
    "WorldModel",
    "WorldModelEnv",
    "Transition",
    "TradingEnv",
    "RiskShapedTradingEnv",
    "DiscreteTradingEnv",
    "HierarchicalTradingEnv",
    "RLLibTradingEnv",
    "MultiAgentTradingEnv",
    "DistributionalAgent",
    "MeanAgent",
]
