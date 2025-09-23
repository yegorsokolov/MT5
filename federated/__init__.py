"""Compatibility stub for the retired federated HTTP helpers.

The historical FastAPI service that backed these modules lives only in the
archived bot API history.  Provide your own implementation (for example by
wrapping ``mt5.remote_api``) if you still need a remote control surface.
"""

raise ImportError(
    "`federated` is preserved solely as a stub. Supply your own federated "
    "client, coordinator, and utility helpers in your deployment."
)
