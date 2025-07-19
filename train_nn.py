"""Train an LSTM model on tick data features."""

from pathlib import Path

import joblib
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


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out).squeeze(1)


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
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    if "SymbolCode" in df.columns:
        features.append("SymbolCode")

    seq_len = cfg.get("sequence_length", 50)
    X_train, y_train = make_sequence_arrays(train_df, features, seq_len)
    X_test, y_test = make_sequence_arrays(test_df, features, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(len(features)).to(device)
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

    joblib.dump(model.state_dict(), root / "model_lstm.pt")
    print("Model saved to", root / "model_lstm.pt")


if __name__ == "__main__":
    main()

