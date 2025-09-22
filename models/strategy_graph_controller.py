"""Neural policy emitting tiny strategy graphs.

The :class:`StrategyGraphController` combines a small graph neural network with
a policy head that selects between a set of pre-defined strategy graphs.  The
network can therefore be trained using REINFORCE style policy gradients where
the reward is the profit and loss (PnL) obtained by executing the generated
graph on historical market data.

This implementation is intentionally lightweight – it is not intended to be a
production ready trading system but rather a concise, testable example used in
the unit tests.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical

from models.graph_net import GraphNet
from strategies.graph_dsl import (
    ExitRule,
    Filter,
    Indicator,
    PositionSizer,
    StrategyGraph,
)


class StrategyGraphController(nn.Module):
    """Graph neural network with a categorical policy head tied to DSL graphs."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden: int = 16,
        actions: int = 2,
        *,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.raw_input_dim = input_dim or in_channels
        self.gnn = GraphNet(
            in_channels=in_channels,
            hidden_channels=hidden,
            out_channels=hidden,
            num_layers=2,
        )
        self.policy = nn.Linear(hidden, actions)
        self.actions = actions
        default_edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.register_buffer("edge_index", default_edge, persistent=False)

    # ------------------------------------------------------------------
    # Feature processing
    def prepare_graph_inputs(
        self,
        features: Sequence[dict] | torch.Tensor | Sequence[Sequence[float]],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return graph node features and a summary for ``features``.

        ``features`` may be a sequence of market data dictionaries, a numpy
        array or any object that :func:`torch.as_tensor` understands.
        """

        node_features, summary = self._summarise_features(features)
        param = next(self.parameters())
        device = device or param.device
        dtype = dtype or param.dtype
        node_features = node_features.to(device=device, dtype=dtype)
        return node_features, summary

    def _summarise_features(
        self, features: Sequence[dict] | torch.Tensor | Sequence[Sequence[float]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(features, torch.Tensor):
            return self._summarise_tensor(features.detach())
        if isinstance(features, Sequence) and features:
            first = features[0]
            if isinstance(first, dict):
                return self._summarise_market_data(features)  # type: ignore[arg-type]
        try:
            tensor = torch.as_tensor(features, dtype=torch.float32)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError("Unsupported feature specification") from exc
        return self._summarise_tensor(tensor)

    def _summarise_market_data(
        self, market_data: Sequence[dict]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not market_data:
            raise ValueError("market_data must not be empty")
        prices = torch.tensor(
            [float(bar.get("price", 0.0)) for bar in market_data], dtype=torch.float32
        )
        mas = torch.tensor(
            [float(bar.get("ma", bar.get("price", 0.0))) for bar in market_data],
            dtype=torch.float32,
        )
        extras: List[float] = []
        for bar in market_data:
            numeric = [
                float(v)
                for key, v in bar.items()
                if key not in {"price", "ma"} and isinstance(v, (int, float))
            ]
            extras.append(sum(numeric) / len(numeric) if numeric else 0.0)
        extra_tensor = torch.tensor(extras, dtype=torch.float32)
        stacked = torch.stack([prices, mas, extra_tensor], dim=1)
        return self._summarise_tensor(stacked)

    def _summarise_tensor(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim > 2:
            tensor = tensor.view(tensor.shape[0], -1)
        if tensor.size(0) == 0:
            raise ValueError("feature tensor must not be empty")
        tensor = tensor.to(torch.float32)
        price = tensor[:, 0]
        ma = tensor[:, 1] if tensor.size(1) > 1 else price
        extras = tensor[:, 2:] if tensor.size(1) > 2 else None

        latest_price = price[-1]
        latest_ma = ma[-1]
        diff = latest_price - latest_ma
        ratio = latest_price / (latest_ma.abs() + 1e-6) - 1.0
        trend = price[-1] - price[0] if price.size(0) > 1 else price.new_tensor(0.0)

        has_macro = bool(
            extras is not None
            and extras.numel() > 0
            and bool(extras.abs().sum().item() > 1e-6)
        )
        macro = extras.mean() if has_macro and extras is not None else tensor.new_tensor(0.0)

        signal = torch.tanh(diff + ratio)
        context = torch.tanh(trend + (macro if has_macro else ratio))
        node_features = torch.stack([signal, context]).view(2, 1)
        summary: Dict[str, float] = {
            "signal": float(signal),
            "diff": float(diff),
            "ratio": float(ratio),
            "trend": float(trend),
            "macro": float(macro),
            "context": float(context),
            "has_macro": has_macro,
        }
        return node_features, summary

    # ------------------------------------------------------------------
    # Model interface
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        if edge_index is None:
            edge_index = self.edge_index
        h = self.gnn(x, edge_index.to(x.device))
        pooled = h.mean(dim=0)
        return self.policy(pooled)

    def sample(
        self, x: torch.Tensor, edge_index: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x, edge_index)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def generate(
        self,
        features: Sequence[dict] | torch.Tensor | Sequence[Sequence[float]],
        risk_profile: float = 0.5,
        *,
        deterministic: bool = True,
        temperature: float = 1.0,
        return_info: bool = False,
    ) -> StrategyGraph | Tuple[StrategyGraph, Dict[str, object]]:
        """Produce a :class:`StrategyGraph` conditioned on ``features``.

        Parameters
        ----------
        features:
            Sequence of market features or raw tensors.
        risk_profile:
            Target risk level between 0 and 1 used to adjust position sizing.
        deterministic:
            When ``True`` choose the highest probability action, otherwise
            sample from the policy distribution.
        temperature:
            Optional softmax temperature applied to the adjusted logits.
        return_info:
            When ``True`` also return metadata about the decision process.
        """

        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        x, summary = self.prepare_graph_inputs(features)
        clamped_risk = self._clamp_risk(risk_profile)
        with torch.no_grad():
            logits = self.forward(x)
            bias = self._feature_bias(
                summary, clamped_risk, logits.dtype, self.actions
            ).to(logits.device)
            adjusted = logits + bias
            if temperature != 1.0:
                adjusted = adjusted / float(temperature)
            if deterministic:
                action = int(torch.argmax(adjusted).item())
            else:
                dist = Categorical(logits=adjusted)
                action = int(dist.sample().item())
        macro = summary["macro"] if summary.get("has_macro") else None
        graph = self.build_graph(action, risk=clamped_risk, macro=macro)
        if not return_info:
            return graph
        info: Dict[str, object] = {
            "action": action,
            "logits": adjusted.detach().cpu(),
            "bias": bias.detach().cpu(),
            "risk": clamped_risk,
            "summary": summary,
        }
        return graph, info

    # ------------------------------------------------------------------
    # Graph helpers
    def build_graph(
        self, action: int, *, risk: float | None = None, macro: float | None = None
    ) -> StrategyGraph:
        """Return a :class:`StrategyGraph` for ``action`` respecting ``risk``."""

        if action == 0:
            indicator = Indicator("price", ">", "ma")
        else:
            indicator = Indicator("price", "<", "ma")
        filt = Filter()
        sizer = PositionSizer(self._risk_to_size(risk, macro))
        exit_rule = ExitRule()
        nodes = {0: indicator, 1: filt, 2: sizer, 3: exit_rule}
        edges = [(0, 1, None), (1, 2, True), (1, 3, False)]
        return StrategyGraph(nodes=nodes, edges=edges)

    # ------------------------------------------------------------------
    # Utilities
    @staticmethod
    def _clamp_risk(risk: float) -> float:
        if not math.isfinite(risk):  # pragma: no cover - defensive
            return 0.5
        return max(0.0, min(1.0, float(risk)))

    @staticmethod
    def _risk_to_size(risk: float | None, macro: float | None) -> float:
        clamped = StrategyGraphController._clamp_risk(risk or 0.5)
        base = 0.5 + clamped
        if macro is None:
            return base
        macro_factor = 1.0 - 0.5 * abs(math.tanh(float(macro)))
        return max(0.1, base * macro_factor)

    @staticmethod
    def _feature_bias(
        summary: Dict[str, float],
        risk: float,
        dtype: torch.dtype,
        num_actions: int,
    ) -> torch.Tensor:
        risk_tensor = torch.tensor(risk, dtype=dtype)
        signal = torch.tensor(summary.get("signal", 0.0), dtype=dtype)
        context = torch.tensor(summary.get("context", 0.0), dtype=dtype)
        macro = torch.tensor(summary.get("macro", 0.0), dtype=dtype)
        directional = torch.tanh(signal + 0.25 * context)
        bias = directional * (0.5 + 0.5 * risk_tensor)
        macro_adjust = torch.tanh(macro) * (0.3 * (1.0 - risk_tensor))
        weights = torch.zeros(num_actions, dtype=dtype)
        if num_actions >= 1:
            weights[0] = bias - macro_adjust
        if num_actions >= 2:
            weights[1] = -bias + macro_adjust
        return weights


def train_strategy_graph_controller(
    data: List[dict],
    episodes: int = 100,
    lr: float = 0.1,
    seed: int = 0,
) -> StrategyGraphController:
    """Train :class:`StrategyGraphController` using policy gradients."""

    torch.manual_seed(seed)
    model = StrategyGraphController(in_channels=1)
    optim = Adam(model.parameters(), lr=lr)
    x, summary = model.prepare_graph_inputs(data)
    macro = summary["macro"] if summary.get("has_macro") else None
    risk = model._clamp_risk(0.5 + 0.25 * abs(summary.get("signal", 0.0)))

    for _ in range(episodes):
        action, log_prob = model.sample(x)
        graph = model.build_graph(int(action.item()), risk=risk, macro=macro)
        reward = graph.run(data)
        loss = -log_prob * reward
        optim.zero_grad()
        loss.backward()
        optim.step()

    return model


__all__ = ["StrategyGraphController", "train_strategy_graph_controller"]

