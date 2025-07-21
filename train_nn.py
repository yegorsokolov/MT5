"""Train a Transformer model on tick data sequences."""

from pathlib import Path

import joblib
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import load_config
from dataset import (
    load_history,
    load_history_from_urls,
    make_features,
    train_test_split,
    make_sequence_arrays,
)
from log_utils import setup_logging, log_exceptions

logger = setup_logging()
from log_utils import setup_logging, log_exceptions


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    """Sequence model using transformer encoder layers."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_symbols: int | None = None,
        emb_dim: int = 8,
    ) -> None:
        super().__init__()
        self.symbol_emb = None
        self.symbol_idx = None
        if num_symbols is not None:
            self.symbol_emb = nn.Embedding(num_symbols, emb_dim)
            self.symbol_idx = input_size - 1  # assume last feature is SymbolCode
            input_size -= 1

        self.input_linear = nn.Linear(input_size + (emb_dim if self.symbol_emb else 0), d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: batch x seq x features
        if self.symbol_emb is not None:
            sym = x[:, :, self.symbol_idx].long()
            x = torch.cat([
                x[:, :, : self.symbol_idx],
                x[:, :, self.symbol_idx + 1 :],
                self.symbol_emb(sym),
            ], dim=-1)

        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1])
        return torch.sigmoid(out).squeeze(1)


@log_exceptions
def main():
    cfg = load_config()
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs = []
    for sym in symbols:
        sym_path = root / "data" / f"{sym}_history.csv"
        if sym_path.exists():
            df_sym = load_history(sym_path)
        else:
            urls = cfg.get("data_urls", {}).get(sym)
            if not urls:
                raise FileNotFoundError(f"No history found for {sym} and no URL configured")
            df_sym = load_history_from_urls(urls)
            df_sym.to_csv(sym_path, index=False)
        df_sym["Symbol"] = sym
        dfs.append(df_sym)

    df = make_features(pd.concat(dfs, ignore_index=True))
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

    train_df, test_df = train_test_split(df, cfg.get("train_rows", len(df) // 2))

    features = [
        "return",
        "ma_5",
        "ma_10",
        "ma_30",
        "ma_60",
        "volatility_30",
        "spread",
        "rsi_14",
        "news_sentiment",
    ]
    features += [
        c
        for c in df.columns
        if c.startswith("cross_corr_")
        or c.startswith("factor_")
        or c.startswith("cross_mom_")
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    if "SymbolCode" in df.columns:
        features.append("SymbolCode")

    seq_len = cfg.get("sequence_length", 50)
    X_train, y_train = make_sequence_arrays(train_df, features, seq_len)
    X_test, y_test = make_sequence_arrays(test_df, features, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_symbols = int(df["Symbol"].nunique()) if "Symbol" in df.columns else None
    model = TransformerModel(
        len(features),
        d_model=cfg.get("d_model", 64),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
        num_symbols=num_symbols,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    for epoch in range(cfg.get("epochs", 5)):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            pred_labels = (preds > 0.5).int()
            correct += (pred_labels == yb.int()).sum().item()
            total += yb.size(0)
    print("Test accuracy:", correct / total)

    joblib.dump(model.state_dict(), root / "model_transformer.pt")
    print("Model saved to", root / "model_transformer.pt")


if __name__ == "__main__":
    main()

