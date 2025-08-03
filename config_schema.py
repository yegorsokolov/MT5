from typing import List
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ConfigSchema(BaseModel):
    """Schema for application configuration."""

    seed: int = Field(42, description="Random seed for reproducibility")
    risk_per_trade: float = Field(..., gt=0, le=1, description="Fraction of capital risked per trade")
    symbols: List[str]

    model_config = ConfigDict(extra="allow")

    @field_validator("symbols")
    @classmethod
    def check_symbols(cls, v: List[str]):
        if not v:
            raise ValueError("symbols must contain at least one symbol")
        return v
