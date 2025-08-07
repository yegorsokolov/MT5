"""Dataset versioning utilities."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def compute_hash(path: Union[str, Path]) -> str:
    """Return the SHA256 hash for the file at ``path``.

    Parameters
    ----------
    path: Union[str, Path]
        File path to hash.
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = ["compute_hash"]
