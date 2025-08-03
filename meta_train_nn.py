"""Meta-train a Transformer model across symbols using MAML."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.nn.utils.stateless import functional_call

from utils import load_config
from dataset import (
    load_history_config,
    make_features,
    train_test_split,
    make_sequence_arrays,
)
from log_utils import setup_logging, log_exceptions


logger = setup_logging()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple operation
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
        num_regimes: int | None = None,
        emb_dim: int = 8,
    ) -> None:
        super().__init__()
        self.symbol_emb = None
        self.symbol_idx = None
        self.regime_emb = None
        self.regime_idx = None

        if num_symbols is not None and num_regimes is not None:
            self.symbol_idx = input_size - 2
            self.regime_idx = input_size - 1
        elif num_symbols is not None:
            self.symbol_idx = input_size - 1
        elif num_regimes is not None:
            self.regime_idx = input_size - 1

        if num_symbols is not None:
            self.symbol_emb = nn.Embedding(num_symbols, emb_dim)
            input_size -= 1
        if num_regimes is not None:
            self.regime_emb = nn.Embedding(num_regimes, emb_dim)
            input_size -= 1

        emb_total = 0
        if self.symbol_emb is not None:
            emb_total += emb_dim
        if self.regime_emb is not None:
            emb_total += emb_dim
        self.input_linear = nn.Linear(input_size + emb_total, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - heavy compute
        if self.symbol_emb is not None and self.regime_emb is not None:
            sym = x[:, :, self.symbol_idx].long()
            reg = x[:, :, self.regime_idx].long()
            base = x[:, :, : self.symbol_idx]
            x = torch.cat([base, self.symbol_emb(sym), self.regime_emb(reg)], dim=-1)
        elif self.symbol_emb is not None:
            sym = x[:, :, self.symbol_idx].long()
            base = x[:, :, : self.symbol_idx]
            x = torch.cat([base, self.symbol_emb(sym)], dim=-1)
        elif self.regime_emb is not None:
            reg = x[:, :, self.regime_idx].long()
            base = x[:, :, : self.regime_idx]
            x = torch.cat([base, self.regime_emb(reg)], dim=-1)

        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1])
        return torch.sigmoid(out).squeeze(1)


def _adapt(
    model: nn.Module,
    weights: Dict[str, torch.Tensor],
    xb: torch.Tensor,
    yb: torch.Tensor,
    loss_fn: nn.Module,
    lr: float,
    steps: int,
) -> Dict[str, torch.Tensor]:
    """Perform inner-loop adaptation and return updated weights."""

    for _ in range(steps):
        preds = functional_call(model, weights, (xb,))
        loss = loss_fn(preds, yb)
        grads = torch.autograd.grad(loss, weights.values(), create_graph=True)
        weights = {
            name: w - lr * g for (name, w), g in zip(weights.items(), grads)
        }
    return weights


def _features(df: pd.DataFrame) -> list[str]:
    feats = [
        "return",
        "ma_5",
        "ma_10",
        "ma_30",
        "ma_60",
        "volatility_30",
        "spread",
        "rsi_14",
        "news_sentiment",
        "market_regime",
    ]
    feats += [
        c
        for c in df.columns
        if c.startswith("cross_corr_")
        or c.startswith("factor_")
        or c.startswith("cross_mom_")
    ]
    if "volume_ratio" in df.columns:
        feats.extend(["volume_ratio", "volume_imbalance"])
    if "SymbolCode" in df.columns:
        feats.append("SymbolCode")
    return feats


@log_exceptions
def main() -> None:  # pragma: no cover - heavy compute
    cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    root = Path(__file__).resolve().parent
    symbols = cfg.get("symbols") or [cfg.get("symbol")]

    dfs = []
    for sym in symbols:
        df_sym = load_history_config(sym, cfg, root)
        df_sym["Symbol"] = sym
        dfs.append(df_sym)

    df = make_features(pd.concat(dfs, ignore_index=True))
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

    features = _features(df)
    seq_len = cfg.get("sequence_length", 50)

    # prepare per-symbol tasks
    tasks: Dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for sym, df_sym in df.groupby("Symbol"):
        train_df, test_df = train_test_split(
            df_sym, cfg.get("train_rows", len(df_sym) // 2)
        )
        X_tr, y_tr = make_sequence_arrays(train_df, features, seq_len)
        X_te, y_te = make_sequence_arrays(test_df, features, seq_len)
        tasks[sym] = (
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
            torch.tensor(X_te, dtype=torch.float32),
            torch.tensor(y_te, dtype=torch.float32),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_symbols = int(df["Symbol"].nunique()) if "Symbol" in df.columns else None
    num_regimes = (
        int(df["market_regime"].nunique()) if "market_regime" in df.columns else None
    )
    model = TransformerModel(
        len(features),
        d_model=cfg.get("d_model", 64),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
        num_symbols=num_symbols,
        num_regimes=num_regimes,
    ).to(device)
    loss_fn = nn.BCELoss()
    meta_opt = torch.optim.Adam(model.parameters(), lr=cfg.get("meta_lr", 1e-3))

    inner_lr = cfg.get("inner_lr", 0.01)
    inner_steps = cfg.get("inner_steps", 1)
    meta_batch = cfg.get("meta_batch_size", len(tasks))

    task_keys = list(tasks.keys())
    for epoch in range(cfg.get("meta_epochs", 5)):
        np.random.shuffle(task_keys)
        for i in range(0, len(task_keys), meta_batch):
            batch = task_keys[i : i + meta_batch]
            meta_opt.zero_grad()
            meta_loss = 0.0
            for sym in batch:
                X_tr, y_tr, X_te, y_te = tasks[sym]
                X_tr, y_tr, X_te, y_te = (
                    X_tr.to(device),
                    y_tr.to(device),
                    X_te.to(device),
                    y_te.to(device),
                )
                weights = {name: p for name, p in model.named_parameters()}
                fast_weights = {name: w.clone() for name, w in weights.items()}
                fast_weights = _adapt(
                    model, fast_weights, X_tr, y_tr, loss_fn, inner_lr, inner_steps
                )
                preds_q = functional_call(model, fast_weights, (X_te,))
                meta_loss += loss_fn(preds_q, y_te)
            meta_loss = meta_loss / len(batch)
            meta_loss.backward()
            meta_opt.step()
        logger.info("Epoch %d meta-loss %.4f", epoch + 1, float(meta_loss))

    out_path = root / "models" / "meta_transformer.pth"
    out_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info("Meta model saved to %s", out_path)

    # Optional evaluation on a new symbol with fine-tuning
    eval_sym = cfg.get("eval_symbol")
    if eval_sym:
        df_new = load_history_config(eval_sym, cfg, root)
        df_new["Symbol"] = eval_sym
        df_new = make_features(df_new)
        if "Symbol" in df_new.columns:
            df_new["SymbolCode"] = 0
        X_all, y_all = make_sequence_arrays(df_new, features, seq_len)
        if len(X_all) == 0:
            return
        X_all = torch.tensor(X_all, dtype=torch.float32)
        y_all = torch.tensor(y_all, dtype=torch.float32)
        split = max(1, int(len(X_all) * 0.1))
        X_adapt, y_adapt = X_all[:split], y_all[:split]
        X_test, y_test = X_all[split:], y_all[split:]

        adapted = TransformerModel(
            len(features),
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            num_symbols=num_symbols,
            num_regimes=num_regimes,
        ).to(device)
        adapted.load_state_dict(model.state_dict())
        opt = torch.optim.Adam(adapted.parameters(), lr=inner_lr)
        for _ in range(cfg.get("finetune_steps", 3)):
            opt.zero_grad()
            preds = adapted(X_adapt.to(device))
            loss = loss_fn(preds, y_adapt.to(device))
            loss.backward()
            opt.step()
        adapted.eval()
        with torch.no_grad():
            preds = adapted(X_test.to(device))
            acc = ((preds > 0.5).float() == y_test.to(device)).float().mean().item()
        logger.info("Fine-tuned accuracy on %s: %.3f", eval_sym, acc)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()

