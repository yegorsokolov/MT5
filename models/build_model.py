from __future__ import annotations

"""Utilities for constructing model architectures scaled to hardware resources."""

from typing import Any, Dict, Mapping

import torch

from utils.resource_monitor import monitor
from models.multi_head import MultiHeadTransformer
from models.graph_net import GraphNet
from models.cross_modal_transformer import CrossModalTransformer
from models.ts_masked_encoder import initialize_model_with_ts_masked_encoder
from models.contrastive_encoder import initialize_model_with_contrastive
from models.ts2vec import initialize_model_with_ts2vec
from models.tft import TemporalFusionTransformer, TFTConfig


def compute_scale_factor(capabilities=None) -> float:
    """Return a scale factor based on available system resources.

    Parameters
    ----------
    capabilities: utils.resource_monitor.ResourceCapabilities, optional
        If not provided the global monitor's capabilities are used.
    """
    caps = capabilities or monitor.capabilities
    if caps.has_gpu or caps.cpus >= 8:
        return 2.0
    if caps.cpus <= 1:
        return 0.5
    return 1.0


def initialize_tft(config: Mapping[str, Any]) -> TemporalFusionTransformer:
    """Return a :class:`TemporalFusionTransformer` built from ``config``.

    Parameters
    ----------
    config:
        Mapping containing model parameters. Expected keys are ``static_size``,
        ``known_size``, ``observed_size``, ``hidden_size``, ``num_heads`` and
        ``quantiles``. Missing values fall back to sensible defaults.
    """

    tft_cfg = TFTConfig(
        static_size=int(config.get("static_size", 0)),
        known_size=int(config.get("known_size", 0)),
        observed_size=int(config.get("observed_size", 0)),
        hidden_size=int(config.get("hidden_size", config.get("d_model", 64))),
        num_heads=int(config.get("num_heads", config.get("nhead", 4))),
        quantiles=config.get("quantiles", (0.1, 0.5, 0.9)),
    )
    return TemporalFusionTransformer(tft_cfg)


def build_model(
    input_size: int,
    cfg: Dict[str, Any],
    scale_factor: float | None = None,
    *,
    num_symbols: int | None = None,
    num_regimes: int | None = None,
) -> torch.nn.Module:
    """Construct a model scaled according to ``scale_factor``.

    ``cfg`` follows the usual training configuration dictionary.  The function
    adjusts ``d_model``, ``nhead`` and ``num_layers`` in proportion to the
    supplied ``scale_factor``.  When ``scale_factor`` is ``None`` it will be
    determined from :mod:`utils.resource_monitor`.
    """

    if scale_factor is None:
        scale_factor = compute_scale_factor()
    d_model = max(1, int(cfg.get("d_model", 64) * scale_factor))
    nhead = max(1, int(cfg.get("nhead", 4) * scale_factor))
    num_layers = max(1, int(cfg.get("num_layers", 2) * scale_factor))

    model_cfg = cfg.get("model", {})
    if model_cfg.get("type") == "cross_modal_transformer":
        news_dim = model_cfg.get("news_dim", input_size)
        model = CrossModalTransformer(
            input_size,
            news_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=cfg.get("dropout", 0.1),
            output_dim=model_cfg.get("output_dim", 1),
        )
    elif cfg.get("graph_model"):
        model = GraphNet(
            input_size,
            hidden_channels=d_model,
            out_channels=1,
            num_layers=num_layers,
        )
    else:
        model = MultiHeadTransformer(
            input_size,
            num_symbols=num_symbols,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_regimes=num_regimes,
            use_checkpointing=cfg.get("use_checkpointing", False),
            dropout=cfg.get("dropout", 0.1),
            ff_dim=cfg.get("ff_dim"),
            layer_norm=cfg.get("layer_norm", False),
            time_encoding=cfg.get("time_encoding", False),
        )

    if cfg.get("use_ts_pretrain"):
        initialize_model_with_ts_masked_encoder(model)
    if cfg.get("use_ts2vec_pretrain"):
        initialize_model_with_ts2vec(model)
    if cfg.get("use_contrastive_pretrain"):
        initialize_model_with_contrastive(model)
    return model
