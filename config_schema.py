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
    log_forward: dict | None = Field(
        None,
        description="Remote log forwarding configuration, e.g. {'url': 'http://host'}",
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("symbols")
    @classmethod
    def check_symbols(cls, v: List[str]):
        if not v:
            raise ValueError("symbols must contain at least one symbol")
        return v
