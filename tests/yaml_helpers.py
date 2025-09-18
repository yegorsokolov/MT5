from __future__ import annotations

import importlib
import sys


def ensure_real_yaml() -> None:
    """Replace the stubbed YAML helpers with the actual PyYAML implementation."""

    stub = sys.modules.get("yaml")
    if stub is None:
        return

    safe_dump = getattr(stub, "safe_dump", None)
    safe_load = getattr(stub, "safe_load", None)
    if getattr(safe_dump, "__name__", "") != "<lambda>" and getattr(
        safe_load, "__name__", ""
    ) != "<lambda>":
        return

    sys.modules.pop("yaml", None)
    real_yaml = importlib.import_module("yaml")
    stub.safe_dump = real_yaml.safe_dump  # type: ignore[attr-defined]
    stub.safe_load = real_yaml.safe_load  # type: ignore[attr-defined]
    sys.modules["yaml"] = stub
