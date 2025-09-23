"""Compatibility stub for the legacy federated client helpers."""

raise ImportError(
    "`federated.client` no longer provides a concrete implementation. "
    "Wire up your own orchestration that coordinates with mt5.remote_api."
)
