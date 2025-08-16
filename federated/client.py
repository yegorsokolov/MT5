"""Federated learning client utilities."""
from __future__ import annotations

import requests
import torch

from state_manager import load_latest_checkpoint, save_checkpoint
from .utils import encode_state_dict, decode_state_dict


class FederatedClient:
    """Simple federated learning client.

    Parameters
    ----------
    server_url:
        Base URL of the coordinator server (``https``).
    api_key:
        API key used for authentication.
    model:
        Model whose ``state_dict`` will be synchronised.
    checkpoint_dir:
        Optional directory for saving checkpoints so training can resume
        from the last completed round.
    """

    def __init__(
        self,
        server_url: str,
        api_key: str,
        model: torch.nn.Module,
        checkpoint_dir: str | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.headers = {"X-API-KEY": api_key}
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.round = 0

        ckpt = load_latest_checkpoint(checkpoint_dir)
        if ckpt:
            self.round, state = ckpt
            model_state = state.get("model")
            if model_state is not None:
                model.load_state_dict(model_state)

    # ------------------------------------------------------------------
    # Communication helpers
    # ------------------------------------------------------------------
    def fetch_global(self) -> None:
        """Retrieve the latest global parameters from the coordinator."""
        resp = requests.get(
            f"{self.server_url}/params",
            headers=self.headers,
            timeout=30,
            verify=True,
        )
        resp.raise_for_status()
        data = resp.json()
        params = data.get("params")
        if params:
            state = decode_state_dict(params)
            self.model.load_state_dict(state)
        self.round = data.get("round", self.round)

    def push_update(self) -> None:
        """Send local parameters to the coordinator and receive the new global
        parameters. The current model state is also persisted locally so
        training can resume if interrupted."""
        state = self.model.state_dict()
        payload = {"round": self.round, "params": encode_state_dict(state)}
        resp = requests.post(
            f"{self.server_url}/update",
            json=payload,
            headers=self.headers,
            timeout=30,
            verify=True,
        )
        resp.raise_for_status()
        data = resp.json()
        params = data.get("params")
        if params:
            new_state = decode_state_dict(params)
            self.model.load_state_dict(new_state)
        self.round = data.get("round", self.round)
        save_checkpoint({"model": state}, self.round, self.checkpoint_dir)
