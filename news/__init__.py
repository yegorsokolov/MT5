"""News utilities package.

The package historically exposes an ``impact_model`` that depends on
``scikit-learn``.  For lightweight environments (such as the unit tests in
this kata) that dependency may be missing.  To keep imports lightweight we
attempt to import the heavy modules lazily and fall back to no-op stubs when
unavailable.
"""

try:  # optional dependency used in some environments
    from .impact_model import get_impact  # type: ignore
except Exception:  # pragma: no cover - optional
    def get_impact(*args, **kwargs):  # type: ignore
        raise RuntimeError("impact model unavailable")

try:  # optional dependency used in some environments
    from . import sentiment_fusion  # type: ignore
except Exception:  # pragma: no cover - optional
    sentiment_fusion = None  # type: ignore
try:  # optional dependency used in some environments
    from .logic_sentiment import LogicSentimentAnalyzer  # type: ignore
except Exception:  # pragma: no cover - optional
    LogicSentimentAnalyzer = None  # type: ignore

__all__ = ["get_impact", "sentiment_fusion", "LogicSentimentAnalyzer"]
