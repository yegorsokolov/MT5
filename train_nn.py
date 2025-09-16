"""Train a Transformer model on tick data sequences."""

import asyncio
import logging
from log_utils import log_exceptions

from pathlib import Path
import random
from typing import Callable, TypeVar, Any

import joblib
import math
from analytics import mlflow_client as mlflow
import numpy as np
import pandas as pd
import torch
import json
import psutil
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from analysis.purged_cv import PurgedTimeSeriesSplit
from tqdm import tqdm
from models import model_store
from models.distillation import distill_teacher_student
from models.build_model import build_model, compute_scale_factor, initialize_tft
from models.hier_forecast import HierarchicalForecaster
from models.quantize import apply_quantization
from models.contrastive_encoder import initialize_model_with_contrastive
from models.slimmable_network import SlimmableNetwork, select_width_multiplier
from analysis.feature_selector import select_features
from models.tft import QuantileLoss
from models.cross_asset_transformer import CrossAssetTransformer
from analysis.prob_calibration import ProbabilityCalibrator, log_reliability
from analysis.active_learning import ActiveLearningQueue, merge_labels
from analysis import model_card
from analysis.domain_adapter import DomainAdapter
from analysis.risk_loss import cvar, max_drawdown, risk_penalty, RiskBudget

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

from utils import load_config
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
from train_utils import prepare_modal_arrays

TIERS = {"lite": 0, "standard": 1, "gpu": 2, "hpc": 3}
T = TypeVar("T")
from data.labels import triple_barrier
import argparse
from ray_utils import (
    init as ray_init,
    shutdown as ray_shutdown,
    cluster_available,
    submit,
)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from core.orchestrator import Orchestrator
from utils.lr_scheduler import LookaheadAdamW
from analysis.grad_monitor import GradientMonitor

logger = logging.getLogger(__name__)

Orchestrator.start()


def _load_donor_state_dict(symbol: str):
    """Return state_dict of latest model trained on a symbol."""
    try:
        versions = model_store.list_versions()
    except Exception:  # pragma: no cover
        return None
    for meta in reversed(versions):
        cfg = meta.get("training_config", {})
        syms = cfg.get("symbols") or [cfg.get("symbol")]
        if syms and symbol in syms:
            state, _ = model_store.load_model(meta["version_id"])
            if isinstance(state, dict):
                return state
    return None


def batch_size_backoff(cfg: dict, train_step: Callable[[int, int], T]) -> T:
    """Run ``train_step`` with automatic batch size backoff.

    The ``train_step`` callable is invoked with ``(batch_size, eval_batch_size)``
    and should raise ``RuntimeError`` or ``torch.cuda.OutOfMemoryError`` on
    memory exhaustion. Batch sizes are halved on OOM until training succeeds or
    ``min_batch_size`` is reached.
    """

    mem_gb = monitor.capabilities.memory_gb
    batch_size = cfg.get("batch_size") or int(mem_gb * 64)
    eval_batch_size = cfg.get("eval_batch_size") or batch_size
    min_batch_size = cfg.get("min_batch_size", 8)
    while True:
        try:
            rss_mb = psutil.Process().memory_info().rss / (1024**2)
            logger.info(
                "Attempting batch_size=%s eval_batch_size=%s rss=%.1fMB mem_gb=%.1f",
                batch_size,
                eval_batch_size,
                rss_mb,
                mem_gb,
            )
            cfg["batch_size"] = batch_size
            cfg["eval_batch_size"] = eval_batch_size
            result = train_step(batch_size, eval_batch_size)
            rss_mb = psutil.Process().memory_info().rss / (1024**2)
            logger.info(
                "Training succeeded with batch_size=%s rss=%.1fMB",
                batch_size,
                rss_mb,
            )
            return result
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            msg = str(exc).lower()
            if "out of memory" in msg and batch_size > min_batch_size:
                rss_mb = psutil.Process().memory_info().rss / (1024**2)
                batch_size = max(batch_size // 2, min_batch_size)
                eval_batch_size = max(eval_batch_size // 2, 1)
                logger.warning(
                    "OOM encountered; reducing batch size to %s (rss=%.1fMB)",
                    batch_size,
                    rss_mb,
                )
                torch.cuda.empty_cache()
            else:
                raise


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
    rank: int = 0,
    world_size: int | None = None,
    cfg: dict | None = None,
    resume_online: bool = False,
    transfer_from: str | None = None,
) -> float:
    if cfg is None:
        cfg = load_config()
    if world_size is None:
        world_size = 1
    tier = cfg.get("capability_tier", "auto")
    if tier == "auto":
        tier = monitor.capabilities.capability_tier()
    if tier in ("lite", "standard"):
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
    torch.set_num_threads(cfg.get("num_threads") or monitor.capabilities.cpus)
    use_cuda = torch.cuda.is_available()
    if world_size > 1:
        backend = "nccl" if use_cuda else "gloo"
        if use_cuda:
            torch.cuda.set_device(rank)
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)
    start_time = datetime.now()

    try:  # avoid starting nested runs
        import mlflow as _mlflow  # type: ignore

        active = _mlflow.active_run()
    except Exception:  # pragma: no cover - mlflow optional
        active = None
    if active is None:
        try:
            mlflow.start_run("training_nn", cfg)
        except Exception:  # pragma: no cover - mlflow optional
            pass
    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs = []
    chunk_size = cfg.get("stream_chunk_size", 100_000)
    stream = cfg.get("stream_history", False)
    for sym in symbols:
        if stream:
            # Stream history in chunks to avoid loading the full dataset into memory
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
    adapter_path = root / "domain_adapter.pkl"
    adapter = DomainAdapter.load(adapter_path)
    num_cols = df.select_dtypes(np.number).columns
    if len(num_cols) > 0:
        adapter.fit_source(df[num_cols])
        df[num_cols] = adapter.transform(df[num_cols])
        adapter.save(adapter_path)
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

    rp = cfg.strategy.risk_profile
    user_budget = RiskBudget(
        max_leverage=rp.leverage_cap, max_drawdown=rp.drawdown_limit
    )
    df["risk_tolerance"] = rp.tolerance
    for name, val in user_budget.as_features().items():
        df[name] = val

    df["tb_label"] = triple_barrier(
        df["mid"],
        cfg.get("pt_mult", 0.01),
        cfg.get("sl_mult", 0.01),
        cfg.get("max_horizon", 10),
    )

        train_df, test_df = train_test_split(df, cfg.get("train_rows", len(df) // 2))

        if cfg.get("use_pseudo_labels"):
            pseudo_dir = root / "data" / "pseudo_labels"
            if pseudo_dir.exists():
                files = list(pseudo_dir.glob("*.parquet")) + list(
                    pseudo_dir.glob("*.csv")
                )
                for p in files:
                    try:
                        if p.suffix == ".parquet":
                            df_pseudo = pd.read_parquet(p)
                        else:
                            df_pseudo = pd.read_csv(p)
                    except Exception:
                        continue
                    if "pseudo_label" not in df_pseudo.columns:
                        continue
                    df_pseudo = df_pseudo.copy()
                    df_pseudo["tb_label"] = df_pseudo["pseudo_label"]
                    train_df = pd.concat(
                        [train_df, df_pseudo], ignore_index=True, sort=False
                    )

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
        features.extend(["risk_tolerance", *user_budget.as_features().keys()])
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

        price_window_cols = sorted(
            c for c in df.columns if c.startswith("price_window_")
        )
        news_emb_cols = sorted(c for c in df.columns if c.startswith("news_emb_"))
        if price_window_cols and news_emb_cols:
            for col in price_window_cols + news_emb_cols:
                if col not in features:
                    features.append(col)

        y_full = df["tb_label"]
        features = select_features(df[features], y_full)
        if "SymbolCode" in features:
            features.remove("SymbolCode")
        use_cross_modal = bool(price_window_cols and news_emb_cols)
        if use_cross_modal:
            for col in price_window_cols + news_emb_cols:
                if col not in features:
                    features.append(col)
        model_cfg = cfg.get("model", {})
        use_tft = model_cfg.get("type") == "tft" and TIERS.get(tier, 0) >= TIERS["gpu"]
        if use_tft:
            known_features = model_cfg.get("known_features", features)
            observed_features = model_cfg.get("observed_features", [])
            static_features = model_cfg.get("static_features", [])
            features = list(
                dict.fromkeys(known_features + observed_features + static_features)
            )
        else:
            known_features = observed_features = static_features = []
        feat_path = root / "selected_features.json"
        feat_path.write_text(json.dumps(features))

        seq_len = cfg.get("sequence_length", 50)
        symbol_codes = (
            sorted(df["SymbolCode"].dropna().unique().astype(int))
            if "SymbolCode" in df.columns
            else [0]
        )
        workers = cfg.get("num_workers") or monitor.capabilities.cpus

        def _preflight(bs: int, _ebs: int) -> None:
            if use_cross_modal:
                torch.empty((bs, 1, 1), dtype=torch.float32, device=device)
            else:
                torch.empty(
                    (bs, seq_len, len(features)), dtype=torch.float32, device=device
                )

        batch_size_backoff(cfg, _preflight)
        batch_size = cfg["batch_size"]
        eval_batch_size = cfg["eval_batch_size"]
        train_loaders: dict[int, DataLoader] = {}
        val_loaders: dict[int, DataLoader] = {}
        test_loaders: dict[int, DataLoader] = {}
        train_samplers: dict[int, DistributedSampler | None] = {}
        X_sample = None
        tasks_list: list[tuple[TensorDataset, TensorDataset]] = []
        price_dim: int | None = None
        news_dim: int | None = None
        modal_price_cols: list[str] = []
        modal_news_cols: list[str] = []
        for code in symbol_codes:
            train_sym = (
                train_df[train_df["SymbolCode"] == code]
                if "SymbolCode" in train_df.columns
                else train_df
            )
            test_sym = (
                test_df[test_df["SymbolCode"] == code]
                if "SymbolCode" in test_df.columns
                else test_df
            )
            if use_cross_modal:
                train_arrays = prepare_modal_arrays(
                    train_sym, train_sym["tb_label"].to_numpy(dtype=np.float32)
                )
                test_arrays = prepare_modal_arrays(
                    test_sym, test_sym["tb_label"].to_numpy(dtype=np.float32)
                )
                if train_arrays is None or test_arrays is None:
                    continue
                price_tr_arr, news_tr_arr, y_tr_arr, _, price_cols_used, news_cols_used = train_arrays
                price_te_arr, news_te_arr, y_te_arr, _, _, _ = test_arrays
                if y_tr_arr is None or y_te_arr is None:
                    continue
                if len(price_tr_arr) <= 1 or len(price_te_arr) == 0:
                    continue
                if price_dim is None:
                    price_dim = price_tr_arr.shape[-1]
                    modal_price_cols = list(price_cols_used)
                if news_dim is None:
                    news_dim = news_tr_arr.shape[-1]
                    modal_news_cols = list(news_cols_used)
                val_size = cfg.get("val_size", 0.2)
                n_splits = max(int(round(1 / val_size)) - 1, 1)
                n_splits = min(n_splits, len(price_tr_arr) - 1) or 1
                splitter = PurgedTimeSeriesSplit(
                    n_splits=n_splits, embargo=cfg.get("max_horizon", 0)
                )
                groups = np.full(len(price_tr_arr), code)
                for tr_idx, va_idx in splitter.split(price_tr_arr, groups=groups):
                    p_tr_fold, p_va_fold = price_tr_arr[tr_idx], price_tr_arr[va_idx]
                    n_tr_fold, n_va_fold = news_tr_arr[tr_idx], news_tr_arr[va_idx]
                    y_tr_fold, y_va_fold = y_tr_arr[tr_idx], y_tr_arr[va_idx]
                price_tr_arr, news_tr_arr, y_tr_arr = p_tr_fold, n_tr_fold, y_tr_fold
                price_va_arr, news_va_arr, y_va_arr = p_va_fold, n_va_fold, y_va_fold
                train_ds = TensorDataset(
                    torch.tensor(price_tr_arr, dtype=torch.float32),
                    torch.tensor(news_tr_arr, dtype=torch.float32),
                    torch.tensor(y_tr_arr, dtype=torch.float32),
                )
                val_ds = TensorDataset(
                    torch.tensor(price_va_arr, dtype=torch.float32),
                    torch.tensor(news_va_arr, dtype=torch.float32),
                    torch.tensor(y_va_arr, dtype=torch.float32),
                )
                test_ds = TensorDataset(
                    torch.tensor(price_te_arr, dtype=torch.float32),
                    torch.tensor(news_te_arr, dtype=torch.float32),
                    torch.tensor(y_te_arr, dtype=torch.float32),
                )
            elif use_tft:
                Xk_tr, y_tr = make_sequence_arrays(
                    train_sym, known_features, seq_len, label_col="tb_label"
                )
                Xk_te, y_te = make_sequence_arrays(
                    test_sym, known_features, seq_len, label_col="tb_label"
                )
                if observed_features:
                    Xo_tr = make_sequence_arrays(
                        train_sym, observed_features, seq_len, label_col="tb_label"
                    )[0]
                    Xo_te = make_sequence_arrays(
                        test_sym, observed_features, seq_len, label_col="tb_label"
                    )[0]
                else:
                    Xo_tr = np.zeros((len(Xk_tr), seq_len, 0), dtype=np.float32)
                    Xo_te = np.zeros((len(Xk_te), seq_len, 0), dtype=np.float32)
                if static_features:
                    Xs_tr = train_sym[static_features].to_numpy()[
                        seq_len - 1 : seq_len - 1 + len(Xk_tr)
                    ]
                    Xs_te = test_sym[static_features].to_numpy()[seq_len - 1 :]
                else:
                    Xs_tr = np.zeros((len(Xk_tr), 0), dtype=np.float32)
                    Xs_te = np.zeros((len(Xk_te), 0), dtype=np.float32)
                if len(Xk_tr) <= 1 or len(Xk_te) == 0:
                    continue
                val_size = cfg.get("val_size", 0.2)
                n_splits = max(int(round(1 / val_size)) - 1, 1)
                n_splits = min(n_splits, len(Xk_tr) - 1) or 1
                splitter = PurgedTimeSeriesSplit(
                    n_splits=n_splits, embargo=cfg.get("max_horizon", 0)
                )
                groups = np.full(len(Xk_tr), code)
                for tr_idx, va_idx in splitter.split(Xk_tr, groups=groups):
                    Xk_tr_fold, y_tr_fold = Xk_tr[tr_idx], y_tr[tr_idx]
                    Xk_va_fold, y_va_fold = Xk_tr[va_idx], y_tr[va_idx]
                    Xo_tr_fold, Xo_va_fold = Xo_tr[tr_idx], Xo_tr[va_idx]
                    Xs_tr_fold, Xs_va_fold = Xs_tr[tr_idx], Xs_tr[va_idx]
                Xk_tr, y_tr = Xk_tr_fold, y_tr_fold
                Xk_va, y_va = Xk_va_fold, y_va_fold
                Xo_tr, Xo_va = Xo_tr_fold, Xo_va_fold
                Xs_tr, Xs_va = Xs_tr_fold, Xs_va_fold
                if X_sample is None and len(Xk_tr) > 0:
                    X_sample = Xk_tr
                train_ds = TensorDataset(
                    torch.tensor(Xk_tr, dtype=torch.float32),
                    torch.tensor(Xs_tr, dtype=torch.float32),
                    torch.tensor(Xo_tr, dtype=torch.float32),
                    torch.tensor(y_tr, dtype=torch.float32),
                )
                val_ds = TensorDataset(
                    torch.tensor(Xk_va, dtype=torch.float32),
                    torch.tensor(Xs_va, dtype=torch.float32),
                    torch.tensor(Xo_va, dtype=torch.float32),
                    torch.tensor(y_va, dtype=torch.float32),
                )
                test_ds = TensorDataset(
                    torch.tensor(Xk_te, dtype=torch.float32),
                    torch.tensor(Xs_te, dtype=torch.float32),
                    torch.tensor(Xo_te, dtype=torch.float32),
                    torch.tensor(y_te, dtype=torch.float32),
                )
            else:
                X_tr, y_tr = make_sequence_arrays(
                    train_sym, features, seq_len, label_col="tb_label"
                )
                X_te, y_te = make_sequence_arrays(
                    test_sym, features, seq_len, label_col="tb_label"
                )
                if len(X_tr) <= 1 or len(X_te) == 0:
                    continue
                val_size = cfg.get("val_size", 0.2)
                n_splits = max(int(round(1 / val_size)) - 1, 1)
                n_splits = min(n_splits, len(X_tr) - 1) or 1
                splitter = PurgedTimeSeriesSplit(
                    n_splits=n_splits, embargo=cfg.get("max_horizon", 0)
                )
                groups = np.full(len(X_tr), code)
                for tr_idx, va_idx in splitter.split(X_tr, groups=groups):
                    X_tr_fold, y_tr_fold = X_tr[tr_idx], y_tr[tr_idx]
                    X_va_fold, y_va_fold = X_tr[va_idx], y_tr[va_idx]
                X_tr, y_tr = X_tr_fold, y_tr_fold
                X_va, y_va = X_va_fold, y_va_fold
                if X_sample is None and len(X_tr) > 0:
                    X_sample = X_tr
                train_ds = TensorDataset(
                    torch.tensor(X_tr, dtype=torch.float32),
                    torch.tensor(y_tr, dtype=torch.float32),
                )
                val_ds = TensorDataset(
                    torch.tensor(X_va, dtype=torch.float32),
                    torch.tensor(y_va, dtype=torch.float32),
                )
                test_ds = TensorDataset(
                    torch.tensor(X_te, dtype=torch.float32),
                    torch.tensor(y_te, dtype=torch.float32),
                )
            sampler = (
                DistributedSampler(
                    train_ds, num_replicas=world_size, rank=rank, shuffle=True
                )
                if world_size > 1
                else None
            )
            train_loaders[code] = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                num_workers=workers,
            )
            val_loaders[code] = DataLoader(
                val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=workers
            )
            test_loaders[code] = DataLoader(
                test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=workers
            )
            train_samplers[code] = sampler
            if not use_cross_modal:
                tasks_list.append((train_ds, val_ds))

        from analysis import meta_learning

        if cfg.get("meta_train") and not use_cross_modal:
            state = meta_learning.meta_train_transformer(
                tasks_list, lambda: meta_learning._LinearModel(len(features))
            )
            meta_learning.save_meta_weights(state, "transformer")
            return 0.0
        if cfg.get("meta_init") and not use_cross_modal:
            from models.meta_learner import steps_to
            from torch.utils.data import TensorDataset

            X_all = torch.tensor(df[features].values, dtype=torch.float32)
            y_all = torch.tensor(
                (df["tb_label"].values).astype(float), dtype=torch.float32
            )
            dataset = TensorDataset(X_all, y_all)
            state = meta_learning.load_meta_weights("transformer")
            new_state, history = meta_learning.fine_tune_model(
                state,
                dataset,
                lambda: meta_learning._LinearModel(len(features)),
            )
            logger.info("Meta-init adaptation steps: %s", steps_to(history))
            meta_learning.save_meta_weights(
                new_state, "transformer", regime=cfg.get("symbol", "asset")
            )
            return 0.0
        if cfg.get("fine_tune") and not use_cross_modal:
            from torch.utils.data import TensorDataset

            regime = (
                int(df["market_regime"].iloc[-1])
                if "market_regime" in df.columns
                else 0
            )
            mask = (
                df["market_regime"] == regime
                if "market_regime" in df.columns
                else np.ones(len(df), dtype=bool)
            )
            X_reg = torch.tensor(df.loc[mask, features].values, dtype=torch.float32)
            y_reg = torch.tensor(
                (df.loc[mask, "tb_label"].values).astype(float), dtype=torch.float32
            )
            dataset = TensorDataset(X_reg, y_reg)
            state = meta_learning.load_meta_weights("transformer")
            new_state, _ = meta_learning.fine_tune_model(
                state,
                dataset,
                lambda: meta_learning._LinearModel(len(features)),
            )
            meta_learning.save_meta_weights(
                new_state, "transformer", regime=f"regime_{regime}"
            )
            return 0.0

        num_symbols = len(symbol_codes)
        num_regimes = (
            int(df["market_regime"].nunique())
            if "market_regime" in df.columns
            else None
        )
        scale_factor = compute_scale_factor()
        architecture_history = [
            {"timestamp": datetime.utcnow().isoformat(), "scale_factor": scale_factor}
        ]

        quantiles = cfg.get("quantiles", [0.1, 0.5, 0.9])
        use_cross_asset = (
            cfg.get("cross_asset_transformer") and TIERS.get(tier, 0) >= TIERS["gpu"]
        )
        horizons = cfg.get("horizons")

        if use_cross_modal and (price_dim is None or news_dim is None):
            logger.warning("Cross-modal inputs unavailable; falling back to default model")
            use_cross_modal = False

        model_input_size = int(price_dim or 1) if use_cross_modal else len(features)

        if use_cross_modal:
            model_cfg = cfg.setdefault("model", {})
            model_cfg["type"] = "cross_modal_transformer"
            model_cfg["price_dim"] = int(price_dim or 1)
            model_cfg["news_dim"] = int(news_dim or 1)
            if modal_price_cols:
                model_cfg.setdefault("price_columns", modal_price_cols)
            if modal_news_cols:
                model_cfg.setdefault("news_columns", modal_news_cols)
            model_cfg.setdefault("time_encoding", cfg.get("time_encoding", False))
            model = build_model(
                model_input_size,
                cfg,
                scale_factor,
                num_symbols=num_symbols,
                num_regimes=num_regimes,
            ).to(device)
        elif horizons and TIERS.get(tier, 0) >= TIERS["standard"]:
            model = HierarchicalForecaster(len(features), horizons).to(device)
        elif use_cross_asset:
            model = CrossAssetTransformer(
                len(features),
                num_symbols or 1,
                d_model=cfg.get("d_model", 64),
                nhead=cfg.get("nhead", 4),
                num_layers=cfg.get("num_layers", 2),
                dim_feedforward=cfg.get("dim_feedforward"),
                dropout=cfg.get("dropout", 0.1),
                layer_norm=cfg.get("layer_norm", False),
                time_encoding=cfg.get("time_encoding", False),
            ).to(device)
        elif use_tft:
            tft_cfg = {
                "static_size": len(static_features),
                "known_size": len(known_features) or len(features),
                "observed_size": len(observed_features),
                "hidden_size": cfg.get("d_model", 64),
                "num_heads": cfg.get("nhead", 4),
                "quantiles": quantiles,
            }
            model = initialize_tft(tft_cfg).to(device)
        elif cfg.get("slimmable"):
            width_multipliers = cfg.get("width_multipliers", [0.25, 0.5, 1.0])
            model = SlimmableNetwork(
                len(features),
                cfg.get("d_model", 64),
                width_multipliers=width_multipliers,
            ).to(device)
            chosen_width = select_width_multiplier(width_multipliers)
            model.set_width(chosen_width)
            architecture_history.append(
                {"timestamp": datetime.utcnow().isoformat(), "width_mult": chosen_width}
            )
            logger.info("Selected width multiplier %s", chosen_width)
        else:
            model = build_model(
                model_input_size,
                cfg,
                scale_factor,
                num_symbols=num_symbols,
                num_regimes=num_regimes,
            ).to(device)
        if cfg.get("use_contrastive_pretrain") and not use_cross_modal:
            model = initialize_model_with_contrastive(model)

        def _watch_model() -> None:
            if isinstance(model, HierarchicalForecaster):
                return
            if isinstance(model, SlimmableNetwork):

                async def _watch() -> None:
                    q = monitor.subscribe()
                    nonlocal model
                    while True:
                        await q.get()
                        new_width = select_width_multiplier(
                            list(model.width_multipliers)
                        )
                        if new_width != model.active_multiplier:
                            model.set_width(new_width)
                            architecture_history.append(
                                {
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "width_mult": new_width,
                                }
                            )
                            logger.info("Adjusted width multiplier to %s", new_width)

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.get_event_loop()
                loop.create_task(_watch())
                return

            async def _watch() -> None:
                q = monitor.subscribe()
                nonlocal model, scale_factor
                while True:
                    await q.get()
                    new_scale = compute_scale_factor()
                    if new_scale != scale_factor:
                        state = model.state_dict()
                        model = build_model(
                            model_input_size,
                            cfg,
                            new_scale,
                            num_symbols=num_symbols,
                            num_regimes=num_regimes,
                        ).to(device)
                        model.load_state_dict(state, strict=False)
                        scale_factor = new_scale
                        architecture_history.append(
                            {
                                "timestamp": datetime.utcnow().isoformat(),
                                "scale_factor": scale_factor,
                            }
                        )
                        logger.info(
                            "Hot-reloaded model with scale factor %s", scale_factor
                        )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            loop.create_task(_watch())

        _watch_model()
        if transfer_from:
            state_dict = _load_donor_state_dict(transfer_from)
            if state_dict:
                model.load_state_dict(state_dict, strict=False)
        if world_size > 1:
            model = DDP(model, device_ids=[rank] if use_cuda else None)
        federated_cfg = cfg.get("federated", {})
        federated_client = None
        if federated_cfg.get("enabled"):
            from federated.client import FederatedClient

            federated_client = FederatedClient(
                federated_cfg["server_url"],
                federated_cfg["api_key"],
                model,
                cfg.get("checkpoint_dir"),
            )
            federated_client.fetch_global()
        optim = LookaheadAdamW(model.parameters(), lr=1e-3)
        grad_monitor = GradientMonitor(
            explode=cfg.get("grad_explode", 1e3),
            vanish=cfg.get("grad_vanish", 1e-6),
            out_dir=root / "reports" / "gradients",
        )
        loss_fn = QuantileLoss(quantiles) if use_tft else torch.nn.BCELoss()

        use_amp = cfg.get("use_amp", False)
        scaler = GradScaler() if use_amp else None

        accumulate_steps = cfg.get("accumulate_steps")
        free_mem_mb = 0.0
        if not accumulate_steps:
            try:
                first_loader = next(iter(train_loaders.values()))
                sample_batch = next(iter(first_loader))
                sample_x = sample_batch[0]
                batch_mem_mb = sample_x.element_size() * sample_x.nelement() / (1024**2)
                if monitor.capabilities.has_gpu and device.type == "cuda":
                    free_mem_mb = torch.cuda.mem_get_info()[0] / (1024**2)
                else:
                    free_mem_mb = psutil.virtual_memory().available / (1024**2)
                accumulate_steps = max(1, int(free_mem_mb / (batch_mem_mb * 2)))
            except Exception:
                accumulate_steps = 1
        cfg["accumulate_steps"] = accumulate_steps
        effective_bs = batch_size * accumulate_steps
        rss_mb = psutil.Process().memory_info().rss / (1024**2)
        logger.info(
            "accumulate_steps=%s effective_batch_size=%s rss=%.1fMB free_mem=%.1fMB",
            accumulate_steps,
            effective_bs,
            rss_mb,
            free_mem_mb,
        )

        patience = cfg.get("patience", 3)
        best_val_loss = float("inf")
        epochs_no_improve = 0
        model_path = root / "model_transformer.pt"

        start_epoch = 0
        start_batch = 0
        ckpt = (
            load_latest_checkpoint(cfg.get("checkpoint_dir"))
            if (rank == 0 and resume_online)
            else None
        )
        if ckpt:
            last_step, state = ckpt
            start_epoch = last_step // steps_per_epoch
            start_batch = last_step % steps_per_epoch
            model.load_state_dict(state["model"])
            optim.load_state_dict(state["optimizer"])
            best_val_loss = state.get("best_val_loss", best_val_loss)
            epochs_no_improve = state.get("epochs_no_improve", 0)
            logger.info(
                "Resuming from checkpoint at step %s (epoch %s batch %s)",
                last_step,
                start_epoch,
                start_batch,
            )

        for epoch in range(start_epoch, cfg.get("epochs", 5)):
            for sampler in train_samplers.values():
                if sampler is not None:
                    sampler.set_epoch(epoch)
            model.train()
            optim.zero_grad()
            iterators = _iter_loaders()
            pbar = tqdm(
                range(steps_per_epoch), desc=f"Epoch {epoch+1}", disable=rank != 0
            )
            for step in pbar:
                losses = []
                batch_returns: list[torch.Tensor] = []
                for code, it in iterators.items():
                    try:
                        batch = next(it)
                    except StopIteration:
                        iterators[code] = iter(train_loaders[code])
                        batch = next(iterators[code])
                    if use_tft:
                        xb, xs, xo, yb = batch
                        xb = xb.to(device)
                        xs = xs.to(device)
                        xo = xo.to(device)
                    elif use_cross_modal:
                        xb_price, xb_news, yb = batch
                        xb_price = xb_price.to(device)
                        xb_news = xb_news.to(device)
                    else:
                        xb, yb = batch
                        xb = xb.to(device)
                    yb = yb.to(device)
                    if use_amp:
                        with autocast():
                            if isinstance(model, HierarchicalForecaster):
                                preds = model(xb)
                                losses.append(loss_fn(preds, yb.float()))
                                batch_returns.append(preds.squeeze())
                            else:
                                if use_tft:
                                    preds = model(xb, static=xs, observed=xo)
                                    losses.append(loss_fn(preds, yb.float()))
                                    batch_returns.append(preds.squeeze())
                                elif use_cross_modal:
                                    preds = model(xb_price, xb_news)
                                    losses.append(loss_fn(preds, yb))
                                    batch_returns.append(preds.squeeze())
                                else:
                                    preds = model(xb, code)
                                    losses.append(loss_fn(preds, yb))
                                    batch_returns.append(preds.squeeze())
                    else:
                        if isinstance(model, HierarchicalForecaster):
                            preds = model(xb)
                            losses.append(loss_fn(preds, yb.float()))
                            batch_returns.append(preds.squeeze())
                        else:
                            if use_tft:
                                preds = model(xb, static=xs, observed=xo)
                                losses.append(loss_fn(preds, yb.float()))
                                batch_returns.append(preds.squeeze())
                            elif use_cross_modal:
                                preds = model(xb_price, xb_news)
                                losses.append(loss_fn(preds, yb))
                                batch_returns.append(preds.squeeze())
                            else:
                                preds = model(xb, code)
                                losses.append(loss_fn(preds, yb))
                                batch_returns.append(preds.squeeze())
                raw_loss = sum(losses) / len(losses)
                # add differentiable risk penalty based on model predictions
                returns_batch = torch.cat(batch_returns)
                penalty = risk_penalty(returns_batch, user_budget)
                loss = raw_loss + penalty
                if use_amp:
                    scaler.scale(loss / accumulate_steps).backward()
                else:
                    (loss / accumulate_steps).backward()
                trend, _ = grad_monitor.track(model.parameters())
                if trend == "explode":
                    for group in optim.param_groups:
                        group["lr"] *= cfg.get("grad_lr_decay", 0.5)
                    if min(g["lr"] for g in optim.param_groups) < cfg.get(
                        "min_lr", 1e-6
                    ):
                        logger.error("Gradient explosion; aborting training")
                        grad_monitor.plot("nn")
                        return 0.0
                elif trend == "vanish":
                    for group in optim.param_groups:
                        group["lr"] *= cfg.get("grad_lr_growth", 2.0)
                if (step + 1) % accumulate_steps == 0 or (step + 1) == steps_per_epoch:
                    if use_amp:
                        scaler.unscale_(optim)
                    max_norm = cfg.get("max_norm", 1.0)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm
                    )
                    logger.debug("Gradient norm: %f", grad_norm)
                    if use_amp:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    optim.zero_grad()
                    if rank == 0:
                        global_step = epoch * steps_per_epoch + step
                        save_checkpoint(
                            {
                                "model": (
                                    model.module.state_dict()
                                    if isinstance(model, DDP)
                                    else model.state_dict()
                                ),
                                "optimizer": optim.state_dict(),
                            },
                            global_step + 1,
                            cfg.get("checkpoint_dir"),
                        )
                        pbar.set_postfix(loss=raw_loss.item())

            if rank == 0:
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                val_returns_eval: list[torch.Tensor] = []
                with torch.no_grad():
                    for code, loader in val_loaders.items():
                        for batch in loader:
                        if use_tft:
                            xb, xs, xo, yb = batch
                            xb = xb.to(device)
                            xs = xs.to(device)
                            xo = xo.to(device)
                        elif use_cross_modal:
                            xb_price, xb_news, yb = batch
                            xb_price = xb_price.to(device)
                            xb_news = xb_news.to(device)
                        else:
                            xb, yb = batch
                            xb = xb.to(device)
                        yb = yb.to(device)
                        with autocast(enabled=use_amp):
                            if isinstance(model, HierarchicalForecaster):
                                preds = model(xb)
                                loss = loss_fn(preds, yb.float())
                                prob = preds[:, 0]
                            else:
                                if use_tft:
                                    preds = model(xb, static=xs, observed=xo)
                                    loss = loss_fn(preds, yb.float())
                                    prob = preds[:, len(quantiles) // 2]
                                elif use_cross_modal:
                                    preds = model(xb_price, xb_news)
                                    loss = loss_fn(preds, yb)
                                    prob = preds.squeeze()
                                else:
                                    preds = model(xb, code)
                                    loss = loss_fn(preds, yb)
                                    prob = preds.squeeze()
                            val_loss += loss.item() * xb.size(0)
                            val_returns_eval.append(prob.detach())
                            if isinstance(model, HierarchicalForecaster):
                                pred_labels = (prob > 0.5).int()
                            else:
                                pred_labels = (
                                    (prob > 0.5).int()
                                    if not use_tft
                                    else (preds[:, len(quantiles) // 2] > 0.5).int()
                                )
                            correct += (pred_labels == yb.int()).sum().item()
                            total += yb.size(0)
                val_loss /= total if total else 1
                val_acc = correct / total if total else 0.0
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                mlflow.log_metric("lr", optim.get_lr(), step=epoch)
                if val_returns_eval:
                    returns_tensor = torch.cat(val_returns_eval)
                    alpha = cfg.get("cvar_level", 0.05)
                    cvar_val = float(-cvar(returns_tensor, alpha))
                    mdd_val = float(max_drawdown(returns_tensor))
                    mlflow.log_metric("val_cvar", cvar_val, step=epoch)
                    mlflow.log_metric("val_max_drawdown", mdd_val, step=epoch)
                    pen = risk_penalty(returns_tensor, user_budget, level=alpha)
                    if pen > 0:
                        logger.warning(
                            "Risk constraints violated at epoch %d: %.4f", epoch + 1, float(pen)
                        )
                        break

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
                    (epoch + 1) * steps_per_epoch,
                    cfg.get("checkpoint_dir"),
                )
                if federated_client is not None:
                    federated_client.push_update()

            if world_size > 1:
                dist.barrier()
            if rank == 0 and epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %s", epoch + 1)
                break

        acc = 0.0
        if rank == 0:
            model.load_state_dict(joblib.load(model_path))
            model.eval()
            calibrator = None
            calib_method = cfg.get("calibration")
            if calib_method:
                val_probs: list[np.ndarray] = []
                val_true: list[np.ndarray] = []
                with torch.no_grad():
                    for code, loader in val_loaders.items():
                        for batch in loader:
                            if use_tft:
                                xb, xs, xo, yb = batch
                                xb = xb.to(device)
                                xs = xs.to(device)
                                xo = xo.to(device)
                            else:
                                xb, yb = batch
                                xb = xb.to(device)
                            yb = yb.to(device)
                            with autocast(enabled=use_amp):
                                if isinstance(model, HierarchicalForecaster):
                                    preds = model(xb)
                                else:
                                    if use_tft:
                                        preds = model(xb, static=xs, observed=xo)
                                    else:
                                        preds = model(xb, code)
                            if isinstance(model, HierarchicalForecaster):
                                prob = preds[:, 0]
                                val_true.append(yb[:, 0].cpu().numpy())
                            else:
                                prob = (
                                    preds[:, len(quantiles) // 2]
                                    if use_tft
                                    else preds.squeeze()
                                )
                                val_true.append(yb.cpu().numpy())
                            val_probs.append(prob.cpu().numpy())
                if val_probs:
                    val_probs_arr = np.concatenate(val_probs)
                    val_true_arr = np.concatenate(val_true)
                    calibrator = ProbabilityCalibrator(method=calib_method).fit(
                        val_true_arr, val_probs_arr
                    )
                    calibrated = calibrator.predict(val_probs_arr)
                    brier_raw, brier_cal = log_reliability(
                        val_true_arr,
                        val_probs_arr,
                        calibrated,
                        root / "reports" / "calibration",
                        "nn",
                        calib_method,
                    )
                    mlflow.log_metric("calibration_brier_raw", brier_raw)
                    mlflow.log_metric("calibration_brier_calibrated", brier_cal)

            correct = 0
            total = 0
            q_metrics = np.zeros(len(quantiles)) if use_tft else None
            with torch.no_grad():
                for code, loader in test_loaders.items():
                    for batch in loader:
                        if use_tft:
                            xb, xs, xo, yb = batch
                            xb = xb.to(device)
                            xs = xs.to(device)
                            xo = xo.to(device)
                        elif use_cross_modal:
                            xb_price, xb_news, yb = batch
                            xb_price = xb_price.to(device)
                            xb_news = xb_news.to(device)
                        else:
                            xb, yb = batch
                            xb = xb.to(device)
                        yb = yb.to(device)
                        with autocast(enabled=use_amp):
                            if isinstance(model, HierarchicalForecaster):
                                preds = model(xb)
                                prob = preds[:, 0]
                            else:
                                if use_tft:
                                    preds = model(xb, static=xs, observed=xo)
                                    prob = preds[:, len(quantiles) // 2]
                                elif use_cross_modal:
                                    preds = model(xb_price, xb_news)
                                    prob = preds.squeeze()
                                else:
                                    preds = model(xb, code)
                                    prob = preds.squeeze()
                        if use_tft:
                            errors = yb.unsqueeze(1) - preds
                            q_tensor = torch.tensor(quantiles, device=errors.device)
                            q_loss = torch.maximum(
                                (q_tensor - 1) * errors, q_tensor * errors
                            )
                            q_metrics += q_loss.sum(0).cpu().numpy()
                        if calibrator is not None:
                            prob_np = prob.cpu().numpy()
                            prob = torch.tensor(
                                calibrator.predict(prob_np), device=prob.device
                            )
                        target = yb[:, 0] if yb.ndim > 1 else yb
                        pred_labels = (prob > 0.5).int()
                        correct += (pred_labels == target.int()).sum().item()
                        total += target.size(0)
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
            if use_tft and q_metrics is not None and total:
                q_metrics /= total
                for i, q in enumerate(quantiles):
                    mlflow.log_metric(f"pinball_q{int(q*100)}", float(q_metrics[i]))
            logger.info("Model saved to %s", model_path)
            mlflow.log_artifact(str(model_path))
            saved_state = joblib.load(model_path)
            base_model = model.module if isinstance(model, DDP) else model
            if isinstance(base_model, SlimmableNetwork):
                version_id = model_store.save_model(
                    saved_state,
                    {**cfg, "width_mult": base_model.active_multiplier},
                    {"val_loss": best_val_loss, "test_accuracy": acc},
                    architecture_history=architecture_history,
                    features=features,
                )
                for mult, state in base_model.export_slices().items():
                    if mult == base_model.active_multiplier:
                        continue
                    slice_path = root / f"model_transformer_{mult}x.pt"
                    joblib.dump(state, slice_path)
                    mlflow.log_artifact(str(slice_path))
                    model_store.save_model(
                        joblib.load(slice_path),
                        {**cfg, "width_mult": mult},
                        {"val_loss": best_val_loss, "test_accuracy": acc},
                        architecture_history=architecture_history,
                        features=features,
                    )
            else:
                version_id = model_store.save_model(
                    saved_state,
                    cfg,
                    {"val_loss": best_val_loss, "test_accuracy": acc},
                    architecture_history=architecture_history,
                    features=features,
                )
            logger.info("Registered model version %s", version_id)
            if use_tft:
                quantile_preds = []
                for _code, loader in test_loaders.items():
                    for batch in loader:
                        xb, xs, xo, _ = batch
                        xb = xb.to(device)
                        xs = xs.to(device)
                        xo = xo.to(device)
                        with torch.no_grad():
                            quantile_preds.append(
                                model(xb, static=xs, observed=xo).cpu()
                            )
                if quantile_preds:
                    q_pred = torch.cat(quantile_preds).numpy()
                    q_cols = [f"q{int(q*100)}" for q in quantiles]
                    q_df = pd.DataFrame(q_pred, columns=q_cols)
                    q_path = root / "tft_quantiles.csv"
                    q_df.to_csv(q_path, index=False)
                    mlflow.log_artifact(str(q_path))
                    logger.info("Saved quantile forecasts to %s", q_path)
                logger.info("TFT variable importance: %s", model.variable_importance())
                if model.last_attention is not None:
                    attn_path = root / "tft_attention.pt"
                    torch.save(model.last_attention.cpu(), attn_path)
                    mlflow.log_artifact(str(attn_path))
                    logger.info("TFT attention weights logged to %s", attn_path)
            else:
                attn_source = model.module if isinstance(model, DDP) else model
                last_attn = getattr(attn_source, "last_attention", None)
                if last_attn is not None:
                    if isinstance(last_attn, dict):
                        cpu_attn = {}
                        for name, tensor in last_attn.items():
                            if tensor is None:
                                continue
                            cpu_tensor = tensor.detach().cpu()
                            cpu_attn[name] = cpu_tensor
                            attn_file = root / f"{name}_attention.pt"
                            torch.save(cpu_tensor, attn_file)
                            try:
                                mlflow.log_artifact(str(attn_file))
                            except Exception:  # pragma: no cover - mlflow optional
                                pass
                            logger.info("Attention weights (%s) logged to %s", name, attn_file)
                        if cpu_attn:
                            attn_path = root / "attention.pt"
                            torch.save(cpu_attn, attn_path)
                            try:
                                mlflow.log_artifact(str(attn_path))
                            except Exception:  # pragma: no cover - mlflow optional
                                pass
                            logger.info("Combined attention weights logged to %s", attn_path)
                    else:
                        attn_path = root / "attention.pt"
                        tensor = last_attn.detach().cpu()
                        torch.save(tensor, attn_path)
                        try:
                            mlflow.log_artifact(str(attn_path))
                        except Exception:  # pragma: no cover - mlflow optional
                            pass
                        logger.info("Attention weights logged to %s", attn_path)
            if cfg.get("quantize"):
                qmodel = apply_quantization(
                    model.module if isinstance(model, DDP) else model
                )
                q_path = root / "model_transformer_quantized.pt"
                joblib.dump(qmodel.state_dict(), q_path)
                model_store.save_model(
                    joblib.load(q_path),
                    {**cfg, "quantized": True},
                    {"val_loss": best_val_loss, "test_accuracy": acc},
                    architecture_history=architecture_history,
                    features=features,
                )
                logger.info("Quantized model saved to %s", q_path)
            student = TransformerModel(
                len(features),
                d_model=max(16, cfg.get("d_model", 64) // 2),
                nhead=max(1, cfg.get("nhead", 4) // 2),
                num_layers=max(1, cfg.get("num_layers", 2) // 2),
                num_symbols=num_symbols,
                num_regimes=num_regimes,
            ).to(device)
            first_loader = next(iter(train_loaders.values()))
            distill_teacher_student(
                model.module if isinstance(model, DDP) else model,
                student,
                first_loader,
                epochs=cfg.get("distill_epochs", 1),
            )
            student_path = root / "model_transformer_distilled.pt"
            joblib.dump(student.state_dict(), student_path)
            model_store.save_model(
                joblib.load(student_path),
                {**cfg, "distilled_from": version_id},
                {"teacher_accuracy": acc},
                architecture_history=architecture_history,
                features=features,
            )
            logger.info("Distilled student model saved to %s", student_path)
            if cfg.get("feature_importance", False) and not use_cross_modal:
                report_dir = root / "reports"
                X_sample_np = (
                    X_sample[: cfg.get("shap_samples", 100)]
                    if X_sample is not None
                    else None
                )
                if X_sample_np is not None:
                    log_shap_importance(
                        model, X_sample_np, features, report_dir, device
                    )

            model_card.generate(
                cfg,
                [root / "data" / f"{s}_history.parquet" for s in symbols],
                features,
                {
                    "best_val_loss": best_val_loss,
                    "val_accuracy": val_acc,
                    "test_accuracy": acc,
                },
                root / "reports" / "model_cards",
            )

    # Active learning: queue uncertain samples and merge any returned labels
    try:
        al_queue = ActiveLearningQueue()
        if not use_cross_modal:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(
                    train_df[features].values, dtype=torch.float32, device=device
                )
                preds = model(X_tensor).cpu().numpy()
            if preds.ndim > 1:
                np.save(root / "pred_short.npy", preds[:, 0])
                np.save(root / "pred_long.npy", preds[:, -1])
                probs = preds[:, 0]
            else:
                np.save(root / "pred_short.npy", preds)
                np.save(root / "pred_long.npy", preds)
                probs = preds
            probs = np.column_stack([1 - probs, probs])
            al_queue.push(train_df.index, probs, k=cfg.get("al_queue_size", 10))
            new_labels = al_queue.pop_labeled()
            if not new_labels.empty:
                train_df = merge_labels(train_df, new_labels, "tb_label")
                save_history_parquet(train_df, root / "data" / "history.parquet")
    except Exception as e:  # pragma: no cover - safety
        logger.warning("Active learning step failed: %s", e)

    if cfg.get("export"):
        from models.export import export_pytorch

        if X_sample is not None and not use_cross_modal:
            sample = torch.tensor(X_sample[:1], dtype=torch.float32)
            export_pytorch(model, sample)

    if world_size > 1:
        dist.destroy_process_group()

    grad_monitor.plot("nn")
    mlflow.log_metric("runtime", (datetime.now() - start_time).total_seconds())
    return acc


def launch(cfg: dict | None = None) -> float:
    if cfg is None:
        cfg = load_config()
    resume_online = cfg.get("resume_online", False)
    transfer_from = cfg.get("transfer_from")
    if cluster_available():
        seeds = cfg.get("seeds", [cfg.get("seed", 42)])
        results = []
        for s in seeds:
            cfg_s = dict(cfg)
            cfg_s["seed"] = s
            results.append(
                submit(
                    main,
                    0,
                    1,
                    cfg_s,
                    resume_online=resume_online,
                    transfer_from=transfer_from,
                )
            )
        return float(results[0] if results else 0.0)
    use_ddp = cfg.get("ddp", monitor.capabilities.ddp())
    world_size = torch.cuda.device_count()
    if use_ddp and world_size > 1:
        mp.spawn(
            main,
            args=(world_size, cfg, resume_online, transfer_from),
            nprocs=world_size,
        )
        return 0.0
    else:
        return main(0, 1, cfg, resume_online=resume_online, transfer_from=transfer_from)


if __name__ == "__main__":
    import warnings

    warnings.warn(
        "train_nn.py is deprecated; use 'python train_cli.py neural' instead",
        DeprecationWarning,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddp", action="store_true", help="Enable DistributedDataParallel"
    )
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search")
    parser.add_argument(
        "--evo-search",
        action="store_true",
        help="Run evolutionary multi-objective parameter search",
    )
    parser.add_argument("--export", action="store_true", help="Export model to ONNX")
    parser.add_argument("--quantize", action="store_true", help="Save quantized model")
    parser.add_argument(
        "--resume-online",
        action="store_true",
        help="Resume incremental training from the latest checkpoint",
    )
    parser.add_argument(
        "--transfer-from",
        type=str,
        help="Initialize model using weights from a donor symbol",
    )
    parser.add_argument(
        "--graph-model",
        action="store_true",
        help="Use GraphNet instead of the transformer",
    )
    parser.add_argument(
        "--meta-train",
        action="store_true",
        help="Run meta-training to produce meta-initialised weights",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune from meta weights on latest regime",
    )
    parser.add_argument(
        "--meta-init",
        action="store_true",
        help="Initialise from meta weights and adapt to current dataset",
    )
    parser.add_argument(
        "--use-pseudo-labels",
        action="store_true",
        help="Include pseudo-labeled samples during training",
    )
    parser.add_argument(
        "--cross-asset-transformer",
        action="store_true",
        help="Enable cross-asset transformer when resources permit",
    )
    args = parser.parse_args()
    cfg = load_config()
    if args.ddp:
        cfg["ddp"] = True
    if args.export:
        cfg["export"] = True
    if args.quantize:
        cfg["quantize"] = True
    if args.resume_online:
        cfg["resume_online"] = True
    if args.transfer_from:
        cfg["transfer_from"] = args.transfer_from
    if args.graph_model:
        cfg["graph_model"] = True
    if args.meta_train:
        cfg["meta_train"] = True
    if args.meta_init:
        cfg["meta_init"] = True
    if args.fine_tune:
        cfg["fine_tune"] = True
    if args.use_pseudo_labels:
        cfg["use_pseudo_labels"] = True
    if args.cross_asset_transformer:
        cfg["cross_asset_transformer"] = True
    if args.tune:
        from tuning.bayesian_search import run_search

        run_search(lambda c, t: launch(c), cfg)
    elif args.evo_search:
        from copy import deepcopy
        from tuning.evolutionary_search import run_evolutionary_search
        from backtest import run_backtest

        def eval_fn(params: dict) -> tuple[float, float, float]:
            trial_cfg = deepcopy(cfg)
            trial_cfg.update(params)
            launch(trial_cfg)
            metrics = run_backtest(trial_cfg)
            return (
                -float(metrics.get("return", 0.0)),
                float(metrics.get("max_drawdown", 0.0)),
                -float(metrics.get("trade_count", metrics.get("trades", 0.0))),
            )

        space = {
            "learning_rate": (1e-5, 1e-2, "log"),
            "num_layers": (1, 4, "int"),
            "d_model": (32, 256, "int"),
        }
        run_evolutionary_search(eval_fn, space)
    else:
        ray_init()
        try:
            launch(cfg)
        finally:
            ray_shutdown()
