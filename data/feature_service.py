"""Stub for the detached feature retrieval API.

The project no longer ships the FastAPI feature service as part of the
installed package.  The historical implementation has been removed entirely,
and this stub raises a clear error so callers know the service has been
retired from the supported surface area.
"""

raise ImportError(
    "The feature retrieval API was removed from the core package. "
    "The FastAPI service has been retired and is no longer distributed."
)
