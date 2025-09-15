from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, ConfigDict


class ConfigError(ValueError):
    """Raised when configuration validation fails."""


class TrainingConfig(BaseModel):
    """Training-related configuration."""

    seed: int = 42
    use_pseudo_labels: bool = False
    drift_method: str = "adwin"
    drift_delta: float = Field(0.002, gt=0)
    drift_threshold: int = Field(3, ge=1)
    drift_cooldown: float = Field(3600.0, ge=0)
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    num_leaves: int | None = None
    learning_rate: float | None = None
    max_depth: int | None = None
    pt_mult: float = Field(0.01, gt=0)
    sl_mult: float = Field(0.01, gt=0)
    max_horizon: int = Field(10, ge=1)
    model_config = ConfigDict(extra="forbid")


class FeaturesConfig(BaseModel):
    """Feature pipeline configuration."""

    latency_threshold: float = 0.0
    features: List[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class StrategyConfig(BaseModel):
    """Trading strategy configuration."""

    symbols: List[str]
    risk_per_trade: float = Field(..., gt=0, le=1)
    session_position_limits: Dict[str, int] = Field(default_factory=dict)
    default_position_limit: int = 1
    use_kalman_smoothing: bool = False
    class RiskProfileConfig(BaseModel):
        """User risk preference profile."""

        tolerance: float = Field(1.0, ge=0.0, description="Risk tolerance multiplier")
        leverage_cap: float = Field(
            1.0, ge=0.0, description="Maximum allowed leverage for positions"
        )
        drawdown_limit: float = Field(
            0.0,
            ge=0.0,
            description="Fractional drawdown at which positions are force-closed",
        )

    risk_profile: RiskProfileConfig = RiskProfileConfig()
    model_config = ConfigDict(extra="forbid")


class ServicesConfig(BaseModel):
    """Configuration for auxiliary services."""

    service_cmds: Dict[str, List[str]] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class AppConfig(BaseModel):
    """Root application configuration grouping all sections."""

    training: TrainingConfig = TrainingConfig()
    features: FeaturesConfig = FeaturesConfig()
    strategy: StrategyConfig
    services: ServicesConfig = ServicesConfig()
    model_config = ConfigDict(extra="allow")

    def get(self, key: str, default=None):
        for section in (self.training, self.features, self.strategy, self.services):
            if key in section.model_fields:
                return getattr(section, key)
        return default

    def update_from(self, other: "AppConfig") -> None:
        """In-place update from another :class:`AppConfig` instance."""

        for field in self.model_fields:
            setattr(self, field, getattr(other, field))
        extra = getattr(other, "__dict__", {})
        for key, value in extra.items():
            if key not in self.model_fields:
                setattr(self, key, value)
