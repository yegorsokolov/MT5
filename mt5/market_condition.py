from __future__ import annotations

"""Aggregate market condition assessment from multiple data sources.

The :class:`MarketConditionAssessor` combines features from a set of
provider callables.  Each provider returns a mapping of feature names to
numeric values describing the current market state.  Providers may gather
data directly from market feeds (prices, order books) or indirectly from
external information such as news or social media sentiment.

Missing optional dependencies are gracefully handled by falling back to
empty feature dictionaries so the assessor remains usable in minimal
installations.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List

FeatureDict = Dict[str, float]


@dataclass
class MarketConditionAssessor:
    """Collect market condition features from multiple providers."""

    providers: List[Callable[[], FeatureDict]] = field(default_factory=list)

    def __post_init__(self) -> None:  # pragma: no cover - simple initialisation
        if not self.providers:
            self.providers = [self._technical_provider, self._news_provider]

    # ------------------------------------------------------------------
    def _technical_provider(self) -> FeatureDict:
        """Return direct market features such as volatility and trend."""
        try:  # optional dependency providing live market features
            from features.technical import current_features  # type: ignore

            data = current_features() or {}
        except Exception:  # pragma: no cover - best effort
            data = {}
        return {
            "volatility": float(data.get("volatility", 0.0)),
            "trend_strength": float(data.get("trend", 0.0)),
            "regime": float(data.get("regime", 0.0)),
        }

    # ------------------------------------------------------------------
    def _news_provider(self) -> FeatureDict:
        """Return indirect signals such as news sentiment."""
        try:  # optional dependency providing sentiment score
            from news.sentiment import latest_sentiment  # type: ignore

            score = latest_sentiment()
        except Exception:  # pragma: no cover - best effort
            score = 0.0
        return {"sentiment": float(score)}

    # ------------------------------------------------------------------
    def assess(self) -> FeatureDict:
        """Return combined market condition features."""
        features: FeatureDict = {}
        for provider in self.providers:
            try:
                features.update(provider())
            except Exception:  # pragma: no cover - ignore failing provider
                continue
        return features


__all__ = ["MarketConditionAssessor"]
