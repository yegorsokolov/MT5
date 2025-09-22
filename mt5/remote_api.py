"""Compatibility wrapper for the remote management API.

The implementation was moved to :mod:`bot_apis.remote_api` to keep all
FastAPI entry points grouped together.  Importing from this module continues to
work so existing scripts and documentation remain valid.
"""
from bot_apis.remote_api import *  # noqa: F401,F403

try:  # pragma: no branch - fallback when ``__all__`` is missing
    from bot_apis.remote_api import __all__ as __all__  # type: ignore  # noqa: F401
except AttributeError:  # pragma: no cover - defensive
    __all__ = [name for name in globals() if not name.startswith("_")]

if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    from bot_apis.remote_api import main

    main()
