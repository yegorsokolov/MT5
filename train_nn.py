"""Train a Transformer model on tick data sequences."""

import logging
from log_utils import setup_logging, log_exceptions

from pathlib import Path
import random

import joblib
import math
import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split as sk_train_test_split
from tqdm import tqdm

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

from utils import load_config, mlflow_run
from utils.resource_monitor import monitor
from state_manager import save_checkpoint, load_latest_checkpoint
from data.history import (
    load_history_parquet,
    save_history_parquet,
    load_history_config,
    load_history_iter,
)
from data.features import (
    make_features,
    train_test_split,
    make_sequence_arrays,
)
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

setup_logging()
logger = logging.getLogger(__name__)

# Periodically refresh hardware capabilities
monitor.start()


class PositionalEncoding(torch.nn.Module):
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


class TransformerModel(torch.nn.Module):
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
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.symbol_emb = None
        self.symbol_idx = None
        self.regime_emb = None
        self.regime_idx = None
        self.use_checkpointing = use_checkpointing

        if num_symbols is not None and num_regimes is not None:
            self.symbol_idx = input_size - 2
            self.regime_idx = input_size - 1
        elif num_symbols is not None:
            self.symbol_idx = input_size - 1
        elif num_regimes is not None:
            self.regime_idx = input_size - 1

        if num_symbols is not None:
            self.symbol_emb = torch.nn.Embedding(num_symbols, emb_dim)
            input_size -= 1
        if num_regimes is not None:
            self.regime_emb = torch.nn.Embedding(num_regimes, emb_dim)
            input_size -= 1

        emb_total = 0
        if self.symbol_emb is not None:
            emb_total += emb_dim
        if self.regime_emb is not None:
            emb_total += emb_dim
        self.input_linear = torch.nn.Linear(input_size + emb_total, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = torch.nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: batch x seq x features
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
        if self.use_checkpointing:
            for layer in self.transformer.layers:
                x = checkpoint(layer, x)
        else:
            x = self.transformer(x)
        out = self.fc(x[:, -1])
        return torch.sigmoid(out).squeeze(1)


def log_shap_importance(
    model: torch.nn.Module,
    X_sample: np.ndarray,
    features: list[str],
    report_dir: Path,
    device: torch.device,
) -> None:
    """Compute SHAP values for the transformer and save a bar plot."""
    if shap is None:
        logger.info("shap not installed, skipping feature importance")
        return
    try:
        import matplotlib.pyplot as plt

        model.eval()
        background = torch.tensor(X_sample, dtype=torch.float32, device=device)
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(background)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_mean = np.abs(shap_values).mean(axis=1)
        report_dir.mkdir(exist_ok=True)
        plt.figure()
        shap.summary_plot(
            shap_mean,
            pd.DataFrame(X_sample.mean(axis=1), columns=features),
            show=False,
            plot_type="bar",
        )
        plt.tight_layout()
        plt.savefig(report_dir / "feature_importance_nn.png")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to compute SHAP values: %s", exc)


@log_exceptions
def main(
    rank: int = 0, world_size: int | None = None, cfg: dict | None = None
) -> float:
    if cfg is None:
        cfg = load_config()
    if world_size is None:
        world_size = 1
    size = cfg.get("model_size", "auto")
    if size == "auto":
        size = monitor.capabilities.model_size()
    if size == "lite":
        cfg.setdefault("d_model", 32)
        cfg.setdefault("num_layers", 1)
    else:
        cfg.setdefault("d_model", 128)
        cfg.setdefault("num_layers", 4)
    seed = cfg.get("seed", 42) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cfg.get("num_threads", 1))
    use_cuda = torch.cuda.is_available()
    if world_size > 1:
        backend = "nccl" if use_cuda else "gloo"
        if use_cuda:
            torch.cuda.set_device(rank)
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)

    with mlflow_run("training_nn", cfg):
        symbols = cfg.get("symbols") or [cfg.get("symbol")]
        dfs = []
        chunk_size = cfg.get("stream_chunk_size", 100_000)
        stream = cfg.get("stream_history", False)
        for sym in symbols:
            if stream:
                pq_path = root / "data" / f"{sym}_history.parquet"
                if pq_path.exists():
                    for chunk in load_history_iter(pq_path, chunk_size):
                        chunk["Symbol"] = sym
                        dfs.append(chunk)
                else:
                    df_sym = load_history_config(
                        sym, cfg, root, validate=cfg.get("validate", False)
                    )
                    df_sym["Symbol"] = sym
                    dfs.append(df_sym)
            else:
                df_sym = load_history_config(
                    sym, cfg, root, validate=cfg.get("validate", False)
                )
                df_sym["Symbol"] = sym
                dfs.append(df_sym)

        df = make_features(
            pd.concat(dfs, ignore_index=True), validate=cfg.get("validate", False)
        )
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
            "market_regime",
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

        X_train, X_val, y_train, y_val = sk_train_test_split(
            X_train,
            y_train,
            test_size=cfg.get("val_size", 0.2),
            random_state=seed,
        )

        if cfg.get("use_data_augmentation", False) or cfg.get(
            "use_diffusion_aug", False
        ):
            fname = (
                "synthetic_sequences_diffusion.npz"
                if cfg.get("use_diffusion_aug", False)
                else "synthetic_sequences.npz"
            )
            aug_path = root / "data" / "augmented" / fname
            if aug_path.exists():
                data = np.load(aug_path)
                X_aug = data["X"]
                y_aug = data["y"]
                X_train = np.concatenate([X_train, X_aug])
                y_train = np.concatenate([y_train, y_aug])

        num_symbols = int(df["Symbol"].nunique()) if "Symbol" in df.columns else None
        num_regimes = (
            int(df["market_regime"].nunique())
            if "market_regime" in df.columns
            else None
        )
        model = TransformerModel(
            len(features),
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            num_symbols=num_symbols,
            num_regimes=num_regimes,
            use_checkpointing=cfg.get("use_checkpointing", False),
        ).to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[rank] if use_cuda else None)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCELoss()

        use_amp = cfg.get("use_amp", False)
        scaler = GradScaler() if use_amp else None

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        )
        train_sampler = (
            DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True
            )
            if world_size > 1
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=128,
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )
        if rank == 0:
            val_loader = DataLoader(val_ds, batch_size=256)
            test_loader = DataLoader(test_ds, batch_size=256)

        patience = cfg.get("patience", 3)
        best_val_loss = float("inf")
        epochs_no_improve = 0
        model_path = root / "model_transformer.pt"

        start_epoch = 0
        ckpt = load_latest_checkpoint(cfg.get("checkpoint_dir")) if rank == 0 else None
        if ckpt:
            last_epoch, state = ckpt
            start_epoch = last_epoch + 1
            model.load_state_dict(state["model"])
            optim.load_state_dict(state["optimizer"])
            best_val_loss = state.get("best_val_loss", best_val_loss)
            epochs_no_improve = state.get("epochs_no_improve", 0)
            logger.info("Resuming from checkpoint at epoch %s", last_epoch)

        for epoch in range(start_epoch, cfg.get("epochs", 5)):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=rank != 0)
            for xb, yb in pbar:
                xb = xb.to(device)
                yb = yb.to(device)
                optim.zero_grad()
                if use_amp:
                    with autocast():
                        preds = model(xb)
                        loss = loss_fn(preds, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    preds = model(xb)
                    loss = loss_fn(preds, yb)
                    loss.backward()
                    optim.step()
                if rank == 0:
                    pbar.set_postfix(loss=loss.item())

            if rank == 0:
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        with autocast(enabled=use_amp):
                            preds = model(xb)
                            loss = loss_fn(preds, yb)
                        val_loss += loss.item() * xb.size(0)
                        pred_labels = (preds > 0.5).int()
                        correct += (pred_labels == yb.int()).sum().item()
                        total += yb.size(0)
                val_loss /= len(val_loader.dataset)
                val_acc = correct / total if total else 0.0
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    joblib.dump(
                        (
                            model.module.state_dict()
                            if isinstance(model, DDP)
                            else model.state_dict()
                        ),
                        model_path,
                    )
                    logger.info("Validation improved; model saved.")
                else:
                    epochs_no_improve += 1

                save_checkpoint(
                    {
                        "model": (
                            model.module.state_dict()
                            if isinstance(model, DDP)
                            else model.state_dict()
                        ),
                        "optimizer": optim.state_dict(),
                        "metrics": {"val_loss": val_loss, "val_accuracy": val_acc},
                        "best_val_loss": best_val_loss,
                        "epochs_no_improve": epochs_no_improve,
                    },
                    epoch,
                    cfg.get("checkpoint_dir"),
                )

            if world_size > 1:
                dist.barrier()
            if rank == 0 and epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %s", epoch + 1)
                break

        acc = 0.0
        if rank == 0:
            model.load_state_dict(joblib.load(model_path))
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    with autocast(enabled=use_amp):
                        preds = model(xb)
                    pred_labels = (preds > 0.5).int()
                    correct += (pred_labels == yb.int()).sum().item()
                    total += yb.size(0)
            acc = correct / total if total else 0.0
            logger.info("Test accuracy: %s", acc)
            logger.info("Best validation loss: %s", best_val_loss)

            mlflow.log_param("epochs", cfg.get("epochs", 5))
            mlflow.log_param("d_model", cfg.get("d_model", 64))
            mlflow.log_param("nhead", cfg.get("nhead", 4))
            mlflow.log_param("num_layers", cfg.get("num_layers", 2))
            mlflow.log_param("patience", cfg.get("patience", 3))
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("best_val_loss", best_val_loss)
            logger.info("Model saved to %s", model_path)
            mlflow.log_artifact(str(model_path))
            if cfg.get("feature_importance", False):
                report_dir = root / "reports"
                X_sample = X_train[: cfg.get("shap_samples", 100)]
                log_shap_importance(model, X_sample, features, report_dir, device)

    if cfg.get("export"):
        from models.export import export_pytorch

        sample = torch.tensor(X_train[:1], dtype=torch.float32)
        export_pytorch(model, sample)

    if world_size > 1:
        dist.destroy_process_group()

    return acc


def launch(cfg: dict | None = None) -> float:
    if cfg is None:
        cfg = load_config()
    use_ddp = cfg.get("ddp", monitor.capabilities.ddp())
    world_size = torch.cuda.device_count()
    if use_ddp and world_size > 1:
        mp.spawn(main, args=(world_size, cfg), nprocs=world_size)
        return 0.0
    else:
        return main(0, 1, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddp", action="store_true", help="Enable DistributedDataParallel"
    )
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search")
    parser.add_argument("--export", action="store_true", help="Export model to ONNX")
    args = parser.parse_args()
    cfg = load_config()
    if args.ddp:
        cfg["ddp"] = True
    if args.export:
        cfg["export"] = True
    if args.tune:
        from tuning.hyperopt import tune_transformer

        tune_transformer(cfg)
    else:
        launch(cfg)
