"""Stub for the detached feature retrieval API.

The project no longer ships the FastAPI feature service as part of the
installed package.  The historical implementation is preserved under
``archive/bot_apis/feature_service.py`` for teams that still need it, but it is
no longer wired into the training code.  Importing this module now raises a
clear error so callers know the service has been removed from the supported
surface area.
"""

raise ImportError(
    "The feature retrieval API was removed from the core package. "
    "Refer to archive/bot_apis/feature_service.py if you still require the "
    "standalone service."
)
