from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, field_validator, ConfigDict


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
