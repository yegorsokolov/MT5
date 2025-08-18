import torch
from typing import Sequence, List

class HierarchicalForecaster(torch.nn.Module):
    """Shared encoder with horizon-specific output heads.

    Parameters
    ----------
    input_size: int
        Number of input features per timestep.
    horizons: Sequence[int]
        Forecast horizons to predict jointly.
    d_model: int, default 64
        Hidden dimension for the shared encoder.
    nhead: int, default 4
        Number of attention heads for the transformer variant.
    num_layers: int, default 2
        Number of layers in the encoder.
    use_transformer: bool, default False
        If ``True`` a transformer encoder is used, otherwise an LSTM.
    """

    def __init__(
        self,
        input_size: int,
        horizons: Sequence[int],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        use_transformer: bool = False,
    ) -> None:
        super().__init__()
        self.horizons = list(horizons)
        self.use_transformer = use_transformer
        if use_transformer:
            self.input_linear = torch.nn.Linear(input_size, d_model)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            feat_dim = d_model
        else:
            self.encoder = torch.nn.LSTM(
                input_size, d_model, num_layers=num_layers, batch_first=True
            )
            feat_dim = d_model
        self.heads = torch.nn.ModuleDict(
            {str(h): torch.nn.Linear(feat_dim, 1) for h in self.horizons}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities for each forecast horizon."""
        if self.use_transformer:
            x = self.input_linear(x)
            enc = self.encoder(x)[:, -1]
        else:
            enc, _ = self.encoder(x)
            enc = enc[:, -1]
        preds: List[torch.Tensor] = []
        for h in self.horizons:
            head = self.heads[str(h)]
            preds.append(torch.sigmoid(head(enc)).squeeze(1))
        return torch.stack(preds, dim=1)

__all__ = ["HierarchicalForecaster"]
