import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Sequence


class SlimmableNetwork(nn.Module):
    """Simple fully connected network supporting width multipliers.

    Parameters
    ----------
    input_dim: int
        Number of input features.
    hidden_dim: int
        Base hidden dimension prior to applying width multipliers.
    output_dim: int, optional
        Number of output features. Defaults to 1.
    width_multipliers: Sequence[float], optional
        Supported width multipliers. Defaults to ``(0.25, 0.5, 1.0)``.

    Notes
    -----
    Call :meth:`set_width` to adjust the active subnetwork before invoking
    :meth:`forward`.  During inference this allows automatic upgrades or
    downgrades as resource probes run.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        width_multipliers: Sequence[float] = (0.25, 0.5, 1.0),
    ) -> None:
        super().__init__()
        self.width_multipliers = sorted(width_multipliers)
        self.hidden_dim = hidden_dim
        self.max_hidden = int(hidden_dim * max(self.width_multipliers))
        # We maintain the full parameter tensors and slice based on width.
        self.fc1_weight = nn.Parameter(torch.randn(self.max_hidden, input_dim) * 0.01)
        self.fc1_bias = nn.Parameter(torch.zeros(self.max_hidden))
        self.fc2_weight = nn.Parameter(torch.randn(output_dim, self.max_hidden) * 0.01)
        self.fc2_bias = nn.Parameter(torch.zeros(output_dim))
        self.active_multiplier = max(self.width_multipliers)

    def set_width(self, width: float) -> None:
        """Select the active subnetwork width multiplier."""

        if width not in self.width_multipliers:
            raise ValueError(f"Unsupported width multiplier {width}")
        self.active_multiplier = width

    def forward(self, x: torch.Tensor, _code: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the network to ``x``.

        The optional ``_code`` argument is ignored but kept for API
        compatibility with other models in the codebase.
        """

        h = int(self.hidden_dim * self.active_multiplier)
        w1 = self.fc1_weight[:h, :]
        b1 = self.fc1_bias[:h]
        x = F.linear(x, w1, b1)
        x = F.relu(x)
        w2 = self.fc2_weight[:, :h]
        b2 = self.fc2_bias
        x = F.linear(x, w2, b2)
        return x

    def export_slices(self) -> Dict[float, Dict[str, torch.Tensor]]:
        """Return state dict slices for each supported width multiplier."""

        slices: Dict[float, Dict[str, torch.Tensor]] = {}
        for m in self.width_multipliers:
            h = int(self.hidden_dim * m)
            slices[m] = {
                "fc1.weight": self.fc1_weight[:h, :].detach().clone(),
                "fc1.bias": self.fc1_bias[:h].detach().clone(),
                "fc2.weight": self.fc2_weight[:, :h].detach().clone(),
                "fc2.bias": self.fc2_bias.detach().clone(),
            }
        return slices
