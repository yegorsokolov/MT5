"""Reinforcement learning utilities."""

from .meta_controller import MetaController, MetaControllerDataset, train_meta_controller

try:  # optional dependencies that may not be available in minimal environments
    from .risk_cvar import CVaRRewardWrapper
except Exception:  # pragma: no cover
    CVaRRewardWrapper = object  # type: ignore

try:
    from .world_model import WorldModel, WorldModelEnv, Transition
except Exception:  # pragma: no cover
    WorldModel = WorldModelEnv = Transition = object  # type: ignore

try:
    from .trading_env import (
        TradingEnv,
        DiscreteTradingEnv,
        HierarchicalTradingEnv,
        RLLibTradingEnv,
    )
except Exception:  # pragma: no cover
    TradingEnv = DiscreteTradingEnv = HierarchicalTradingEnv = RLLibTradingEnv = object  # type: ignore

try:
    from .risk_shaped_env import RiskShapedTradingEnv
except Exception:  # pragma: no cover
    RiskShapedTradingEnv = object  # type: ignore

try:
    from .macro_reward_wrapper import MacroRewardWrapper
except Exception:  # pragma: no cover
    MacroRewardWrapper = object  # type: ignore

try:
    from .multi_agent_env import MultiAgentTradingEnv
except Exception:  # pragma: no cover
    MultiAgentTradingEnv = object  # type: ignore

try:
    from .distributional_agent import DistributionalAgent, MeanAgent
except Exception:  # pragma: no cover
    DistributionalAgent = MeanAgent = object  # type: ignore

__all__ = [
    "CVaRRewardWrapper",
    "WorldModel",
    "WorldModelEnv",
    "Transition",
    "TradingEnv",
    "RiskShapedTradingEnv",
    "MacroRewardWrapper",
    "DiscreteTradingEnv",
    "HierarchicalTradingEnv",
    "RLLibTradingEnv",
    "MultiAgentTradingEnv",
    "DistributionalAgent",
    "MeanAgent",
    "MetaController",
    "MetaControllerDataset",
    "train_meta_controller",
]
