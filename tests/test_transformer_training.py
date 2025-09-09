import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SimpleTransformer(torch.nn.Module):
    def __init__(self, input_size: int = 1) -> None:
        super().__init__()
        d_model = 16
        nhead = 2
        self.input_linear = torch.nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.fc = torch.nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1])
        return torch.sigmoid(out).squeeze(1)


def test_simple_transformer_trains():
    rng = np.random.default_rng(0)
    n_samples = 200
    seq_len = 5
    X = torch.tensor(rng.standard_normal((n_samples, seq_len, 1)), dtype=torch.float32)
    y = (X.mean(dim=1).squeeze() > 0).float()

    model = SimpleTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()

    losses = []
    for _ in range(100):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[0] > losses[-1] * 2

