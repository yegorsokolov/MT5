"""Compatibility wrapper for the feature retrieval API."""
from bot_apis.feature_service import *  # noqa: F401,F403

try:  # pragma: no branch - graceful when the target omits ``__all__``
    from bot_apis.feature_service import __all__ as __all__  # type: ignore  # noqa: F401
except AttributeError:  # pragma: no cover - defensive
    __all__ = [name for name in globals() if not name.startswith("_")]
