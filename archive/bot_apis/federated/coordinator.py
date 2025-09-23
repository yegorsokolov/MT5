"""Coordinator server for federated learning."""
from __future__ import annotations

from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException, Request

from .utils import encode_state_dict, decode_state_dict


class Coordinator:
    """Round-based parameter averaging server.

    The server collects model updates from multiple clients, averages the
    parameters once all expected clients have submitted, and responds with
    the aggregated parameters. Communication is secured using HTTPS and
    an API key supplied via the ``X-API-KEY`` header.
    """

    def __init__(self, num_clients: int, api_key: str) -> None:
        self.num_clients = num_clients
        self.api_key = api_key
        self.app = FastAPI()
        self.current_round = 0
        self.global_state: Dict[str, torch.Tensor] | None = None
        self.updates: List[Dict[str, torch.Tensor]] = []

        @self.app.get("/params")
        async def get_params(request: Request) -> Dict[str, str | int | None]:
            self._authenticate(request)
            encoded = (
                encode_state_dict(self.global_state) if self.global_state else None
            )
            return {"round": self.current_round, "params": encoded}

        @self.app.post("/update")
        async def post_update(request: Request) -> Dict[str, str | int | None]:
            self._authenticate(request)
            payload = await request.json()
            state = decode_state_dict(payload["params"])
            self.updates.append(state)
            if len(self.updates) >= self.num_clients:
                self.global_state = self._average(self.updates)
                self.updates.clear()
                self.current_round += 1
            encoded = (
                encode_state_dict(self.global_state) if self.global_state else None
            )
            return {"round": self.current_round, "params": encoded}

    # ------------------------------------------------------------------
    def _authenticate(self, request: Request) -> None:
        key = request.headers.get("X-API-KEY")
        if key != self.api_key:
            raise HTTPException(status_code=403, detail="Invalid API key")

    def _average(self, states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        avg = {}
        for k in states[0].keys():
            tensors = [s[k] for s in states]
            stacked = torch.stack([t.detach().float() for t in tensors])
            avg[k] = stacked.mean(dim=0)
        return avg

    def run(self, host: str = "0.0.0.0", port: int = 8443, certfile: str | None = None, keyfile: str | None = None) -> None:
        """Start the coordinator server using ``uvicorn``."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            ssl_certfile=certfile,
            ssl_keyfile=keyfile,
        )
