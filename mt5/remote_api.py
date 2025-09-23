"""Stub for the retired remote management API.

The FastAPI application that previously exposed lifecycle management endpoints
now lives outside of the installable package.  The archived implementation is
kept under ``archive/bot_apis/remote_api.py`` for reference, but the codebase no
longer wires it into the supported workflows.  Importing this module therefore
fails fast with an informative error.
"""

raise ImportError(
    "The remote management API has been removed from the MT5 package. "
    "Refer to archive/bot_apis/remote_api.py for the preserved source if you "
    "still need to host it separately."
)
