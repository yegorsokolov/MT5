from __future__ import annotations

from typing import Any, Dict, Iterable, List


class LiveTradingEnv:
    """Minimal environment storing a stream of feature dictionaries."""

    def __init__(self) -> None:
        self.stream: List[Dict[str, float]] = []

    def append_live_features(self, features: Dict[str, float]) -> None:
        self.stream.append(features)


def incremental_policy_update(
    env: "LiveTradingEnv", model: Any, feature_stream: Iterable[Dict[str, float]], update_steps: int = 1
) -> None:
    """Consume a feature stream and update ``model`` incrementally."""

    for chunk in feature_stream:
        env.append_live_features(chunk)
        if hasattr(model, "learn"):
            model.learn(total_timesteps=update_steps, reset_num_timesteps=False)


__all__ = ["LiveTradingEnv", "incremental_policy_update"]
