"""Stub for the detached feature retrieval API.

This placeholder remains so legacy imports fail fast with a helpful error.
Teams that require an HTTP surface should provide their own shim that wraps
``mt5.remote_api`` from their deployment environment rather than importing
this stub.
"""

raise ImportError(
    "`data.feature_service` is only kept as a compatibility stub. "
    "Expose feature retrieval through your own service that delegates to "
    "mt5.remote_api instead."
)
