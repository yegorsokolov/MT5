from __future__ import annotations

from typing import Any, List
from pydantic import BaseModel, Field, field_validator, ConfigDict


class BaseModelConfig(BaseModel):
    """Configuration for a single ensemble component model."""

    type: str = Field(
        ...,
        description="Model type, e.g. lightgbm, cross_asset_transformer, neural_quantile",
    )
    params: dict[str, Any] | None = Field(
        None, description="Optional model hyper-parameters"
    )
    features: List[str] | None = Field(
        None, description="Optional subset of feature columns to use"
    )


class EnsembleConfig(BaseModel):
    """Configuration for ensemble training."""

    enabled: bool = Field(False, description="Enable training of an ensemble model")
    base_models: dict[str, BaseModelConfig] = Field(
        default_factory=dict,
        description="Mapping of model name to its configuration",
    )
    meta_learner: bool = Field(
        False, description="Train LogisticRegression stacking meta learner"
    )
    diversity_weighting: bool = Field(
        False, description="Weight models inversely to error correlations"
    )


class ConfigSchema(BaseModel):
    """Schema for application configuration."""

    seed: int = Field(42, description="Random seed for reproducibility")
    risk_per_trade: float = Field(
        ..., gt=0, le=1, description="Fraction of capital risked per trade"
    )
    symbols: List[str] = Field(
        ..., description="List of trading symbols, e.g. ['EURUSD', 'GBPUSD']"
    )
    ddp: bool | None = Field(
        None, description="Enable DistributedDataParallel if true, auto-detect if null"
    )
    balance_classes: bool = Field(
        False, description="Apply class weighting during model training"
    )
    time_decay_half_life: int | None = Field(
        None, ge=1, description="Half-life in bars for time-decay sample weighting"
    )
    pred_cache_size: int = Field(
        256, ge=0, description="Maximum entries for prediction cache"
    )
    pred_cache_policy: str = Field(
        "lru", description="Eviction policy for prediction cache (lru or fifo)"
    )
    plugin_cache_ttl: float = Field(
        0,
        ge=0,
        description="Seconds before unused plugins are unloaded from memory",
    )
    drift_method: str = Field("adwin", description="Concept drift detection method")
    drift_delta: float = Field(
        0.002,
        gt=0,
        description="Sensitivity parameter for concept drift detector",
    )
    service_cmds: dict[str, list[str]] | None = Field(
        None,
        description="Mapping of service name to command used to (re)start it",
    )
    ewma_alpha: float = Field(
        0.06,
        gt=0,
        le=1,
        description="Smoothing factor for EWMA risk metrics",
    )
    log_forward: dict | None = Field(
        None,
        description="Remote log forwarding configuration, e.g. {'url': 'http://host'}",
    )
    mlflow: MLflowConfig | None = Field(
        None,
        description="MLflow tracking configuration including remote server credentials",
    )
    ensemble: EnsembleConfig | None = Field(
        None, description="Ensemble training configuration"
    )
    graph_model: bool = Field(
        False, description="Use graph neural network architecture for training"
    )

    use_ts_pretrain: bool = Field(
        False,
        description="Initialise models from time-series masked encoder weights if available",
    )
    ts_pretrain_epochs: int = Field(
        20,
        ge=1,
        description="Number of epochs for time-series encoder pretraining",
    )
    ts_pretrain_batch_size: int = Field(
        32,
        ge=1,
        description="Batch size for time-series encoder pretraining",
    )

    use_contrastive_pretrain: bool = Field(
        False,
        description="Initialise models from contrastive encoder weights if available",
    )
    contrastive_epochs: int = Field(
        20,
        ge=1,
        description="Number of epochs for contrastive encoder pretraining",
    )
    contrastive_batch_size: int = Field(
        32,
        ge=1,
        description="Batch size for contrastive encoder pretraining",
    )

    class CrossAssetConfig(BaseModel):
        window: int | None = Field(30, ge=1, description="Rolling correlation window")
        whitelist: List[str] | None = Field(
            None, description="Subset of symbols for pairwise calculations"
        )
        max_pairs: int | None = Field(
            None, ge=1, description="Maximum number of symbol pairs to retain"
        )
        reduce: str | None = Field(
            "top_k", description="Reduction strategy", pattern="^(top_k|pca)$"
        )

    cross_asset: CrossAssetConfig | None = Field(
        None, description="Cross-asset feature options"
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("symbols")
    @classmethod
    def check_symbols(cls, v: List[str]):
        if not v:
            raise ValueError("symbols must contain at least one symbol")
        return v


class MLflowConfig(BaseModel):
    """Optional MLflow tracking server configuration."""

    tracking_uri: str | None = Field(
        None, description="Remote MLflow tracking URI, e.g. http://mlflow:5000"
    )
    username: str | None = Field(None, description="MLflow tracking server username")
    password: str | None = Field(None, description="MLflow tracking server password")
