"""News impact model utilities."""

from .impact_model import get_impact

try:  # optional dependency used in some environments
    from . import sentiment_fusion  # type: ignore
except Exception:  # pragma: no cover - optional
    sentiment_fusion = None  # type: ignore

__all__ = ["get_impact", "sentiment_fusion"]
