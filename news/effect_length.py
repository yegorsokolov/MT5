"""Utilities to estimate how long a news event is likely to matter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

__all__ = ["EffectLength", "estimate_effect_length"]


@dataclass(frozen=True)
class EffectLength:
    """Container describing the expected persistence of a news item."""

    minutes: float
    """Estimated duration (in minutes) that the headline meaningfully influences price."""

    half_life: float
    """Estimated half-life in minutes before the impact decays materially."""

    score: float
    """Normalised persistence score in ``[0, 1]`` for modelling convenience."""

    importance: float
    """Importance proxy in ``[0, 1]`` inferred from headline context."""


# Keywords that typically imply multi-session or structural effects where the
# market tends to re-price fundamentals or positioning over a longer window.
_STRUCTURAL_TERMS = {
    "merger",
    "acquisition",
    "buyback",
    "dividend",
    "split",
    "spinoff",
    "spin-off",
    "guidance",
    "outlook",
    "forecast",
    "long-term",
    "multi-year",
    "contract",
    "partnership",
    "expansion",
    "investment",
    "regulation",
    "approval",
    "sanction",
    "tariff",
    "ban",
    "lawsuit",
    "investigation",
    "antitrust",
    "settlement",
    "bankruptcy",
    "restructuring",
    "stimulus",
}

# Terms that often keep the market's focus for several sessions (e.g. guidance,
# analyst actions, major product cycles).
_MEDIUM_TERM_TERMS = {
    "earnings",
    "results",
    "quarter",
    "guidance",
    "target",
    "downgrade",
    "upgrade",
    "rating",
    "estimate",
    "outlook",
    "forecast",
    "revenue",
    "sales",
    "production",
    "supply",
    "demand",
    "margin",
    "profit",
    "loss",
    "ipo",
    "backlog",
    "pipeline",
}

# Words that usually describe fleeting, intraday catalysts where the market
# response decays quickly once the headline is digested.
_SHORT_LIVED_TERMS = {
    "preview",
    "recap",
    "brief",
    "tweet",
    "rumor",
    "rumour",
    "commentary",
    "opinion",
    "note",
    "watch",
    "blog",
    "snippet",
    "update",
    "headline",
}


def _keyword_profile(text: str) -> Tuple[float, float]:
    """Return (importance_hint, duration_bias) derived from keywords."""

    lowered = text.lower()
    structural = any(term in lowered for term in _STRUCTURAL_TERMS)
    medium = any(term in lowered for term in _MEDIUM_TERM_TERMS)
    fleeting = any(term in lowered for term in _SHORT_LIVED_TERMS)

    importance = 0.3
    duration_bias = 0.0

    if structural:
        importance += 0.4
        duration_bias += 0.6
    if medium:
        importance += 0.2
        duration_bias += 0.25
    if fleeting:
        importance -= 0.2
        duration_bias -= 0.35

    importance = max(0.05, min(1.0, importance))
    duration_bias = max(-0.6, min(1.0, duration_bias))
    return importance, duration_bias


def estimate_effect_length(
    text: str,
    magnitude: float,
    *,
    importance: float | None = None,
    risk_scale: float = 1.0,
) -> EffectLength:
    """Estimate how long a headline's effect is likely to persist."""

    try:
        mag = abs(float(magnitude))
    except (TypeError, ValueError):
        mag = 0.0
    mag = max(0.0, min(1.0, mag))

    importance_hint, duration_bias = _keyword_profile(text or "")
    if importance is not None:
        try:
            importance_val = float(importance)
        except (TypeError, ValueError):
            importance_val = 0.0
        importance_val = max(0.0, min(1.0, importance_val))
        used_importance = max(importance_val, importance_hint)
    else:
        used_importance = importance_hint

    try:
        risk = float(risk_scale)
    except (TypeError, ValueError):
        risk = 1.0
    risk = max(0.3, min(1.7, risk))

    base_minutes = 45.0 + 110.0 * used_importance
    magnitude_minutes = 180.0 * mag
    keyword_multiplier = 1.0 + 0.35 * duration_bias
    keyword_multiplier = max(0.6, min(1.8, keyword_multiplier))
    risk_multiplier = 0.85 + 0.3 * risk
    risk_multiplier = max(0.7, min(1.4, risk_multiplier))

    effect_minutes = (base_minutes + magnitude_minutes) * keyword_multiplier * risk_multiplier
    effect_minutes = float(max(15.0, min(720.0, effect_minutes)))

    half_life = effect_minutes * (0.35 + 0.45 * used_importance + 0.2 * mag)
    half_life = float(max(10.0, min(720.0, half_life)))

    score = math.tanh(effect_minutes / 180.0)
    score = float(max(0.0, min(1.0, score)))

    return EffectLength(
        minutes=effect_minutes,
        half_life=half_life,
        score=score,
        importance=used_importance,
    )
