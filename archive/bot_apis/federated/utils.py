"""Helper utilities for serialising model parameters."""
from __future__ import annotations

import base64
import io
from typing import Dict

import torch


def encode_state_dict(state: Dict[str, torch.Tensor]) -> str:
    """Return base64 encoded ``state``.

    Parameters
    ----------
    state:
        A ``state_dict`` mapping tensor names to ``Tensor`` objects.
    """
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_state_dict(data: str) -> Dict[str, torch.Tensor]:
    """Decode base64 string to a ``state_dict``."""
    buffer = io.BytesIO(base64.b64decode(data.encode("utf-8")))
    return torch.load(buffer, map_location="cpu")
