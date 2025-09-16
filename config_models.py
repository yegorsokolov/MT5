from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ConfigError(ValueError):
    """Raised when configuration validation fails."""


class DriftConfig(BaseModel):
    """Configuration for the concept drift detector."""

    method: str = Field("adwin", description="Concept drift detection method")
    delta: float = Field(0.002, gt=0, description="Sensitivity of the detector")
    threshold: int = Field(
        3,
        ge=1,
        description="Number of drift events required before reacting",
    )
    cooldown: float = Field(
        3600.0, ge=0.0, description="Seconds to wait before reacting again"
    )
    model_config = ConfigDict(extra="forbid")


class ActiveLearningConfig(BaseModel):
    """Settings controlling the active learning sampling loop."""

    k: int = Field(10, ge=1, description="Number of candidates to query")
    min_confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Skip examples whose model confidence falls below the threshold",
    )
    exploration_bias: float = Field(
        0.0,
        ge=0.0,
        description="Weight given to sampling unexplored regions",
    )
    model_config = ConfigDict(extra="forbid")


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""

    tracking_uri: str = Field(
        "",
        description="Remote MLflow tracking URI; empty string keeps local logging",
    )
    username: str = Field("", description="MLflow username when basic auth is enabled")
    password: str = Field("", description="MLflow password for basic auth")
    model_config = ConfigDict(extra="forbid")


class SMTPConfig(BaseModel):
    """SMTP credentials used by :class:`AlertingConfig`."""

    host: str = ""
    port: int = Field(587, ge=0, le=65535)
    username: str = ""
    password: str = ""
    sender: str = Field("", alias="from")
    recipients: List[str] = Field(default_factory=list, alias="to")
    use_tls: bool = True
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("recipients", mode="before")
    @classmethod
    def _coerce_recipients(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            recipients = [str(v) for v in value if str(v)]
            return recipients
        raise TypeError("Recipients must be provided as a list or comma separated string")


class AlertingConfig(BaseModel):
    """Configuration for alert routing."""

    slack_webhook: str | None = Field(
        None, description="Slack webhook URL used for alert delivery"
    )
    smtp: SMTPConfig | None = None
    model_config = ConfigDict(extra="forbid")


class TrainingConfig(BaseModel):
    """Training-related configuration."""

    seed: int = 42
    model_type: str = Field(
        "lgbm",
        description="Primary model architecture to train (lgbm, neural, cross_modal)",
    )
    use_pseudo_labels: bool = False
    use_focal_loss: bool = False
    focal_alpha: float = Field(0.25, ge=0.0)
    focal_gamma: float = Field(2.0, ge=0.0)
    num_leaves: int | None = Field(None, ge=2)
    learning_rate: float | None = Field(None, gt=0)
    max_depth: int | None = Field(None, ge=1)
    pt_mult: float = Field(0.01, gt=0)
    sl_mult: float = Field(0.01, gt=0)
    max_horizon: int = Field(10, ge=1)
    balance_classes: bool = False
    time_decay_half_life: int | None = Field(None, ge=1)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _merge_legacy_drift(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        data = dict(values)
        drift_fields: Dict[str, Any] = {}
        for key in list(data.keys()):
            if key.startswith("drift_"):
                drift_fields[key.split("drift_", 1)[1]] = data.pop(key)
        if drift_fields and "drift" not in data:
            data["drift"] = drift_fields
        return data

    @field_validator("model_type", mode="before")
    @classmethod
    def _validate_model_type(cls, value: Any) -> str:
        if value is None:
            return "lgbm"
        mt = str(value).lower()
        allowed = {"lgbm", "neural", "cross_modal"}
        if mt not in allowed:
            raise ValueError(
                f"model_type must be one of {sorted(allowed)}, received '{value}'"
            )
        return mt


class FeaturesConfig(BaseModel):
    """Feature pipeline configuration."""

    latency_threshold: float = Field(0.0, ge=0.0)
    features: List[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")

    @field_validator("features", mode="after")
    @classmethod
    def _ensure_unique(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for feature in value:
            if feature not in seen:
                seen.add(feature)
                unique.append(feature)
        return unique


class StrategyConfig(BaseModel):
    """Trading strategy configuration."""

    symbols: List[str]
    risk_per_trade: float = Field(..., gt=0, le=1)
    session_position_limits: Dict[str, int] = Field(default_factory=dict)
    default_position_limit: int = Field(1, ge=0)
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

        model_config = ConfigDict(extra="forbid")

    risk_profile: RiskProfileConfig = RiskProfileConfig()
    model_config = ConfigDict(extra="forbid")

    @field_validator("symbols")
    @classmethod
    def _require_symbols(cls, value: Sequence[str]) -> list[str]:
        symbols = [str(v) for v in value if str(v)]
        if not symbols:
            raise ValueError("At least one trading symbol must be provided")
        return symbols

    @field_validator("session_position_limits", mode="before")
    @classmethod
    def _normalise_limits(cls, value: Any) -> dict[str, int]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("session_position_limits must be a mapping")
        limits: dict[str, int] = {}
        for key, val in value.items():
            limit = int(val)
            if limit < 0:
                raise ValueError("Position limits must be non-negative")
            limits[str(key)] = limit
        return limits


class ServicesConfig(BaseModel):
    """Configuration for auxiliary services."""

    service_cmds: Dict[str, List[str]] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")

    @field_validator("service_cmds", mode="before")
    @classmethod
    def _validate_service_cmds(cls, value: Any) -> dict[str, list[str]]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("service_cmds must be a mapping")
        commands: dict[str, list[str]] = {}
        for name, cmd in value.items():
            if isinstance(cmd, (str, bytes)):
                raise ValueError(
                    f"Service '{name}' command must be provided as a sequence of arguments"
                )
            if not isinstance(cmd, Sequence):
                raise TypeError(
                    f"Service '{name}' command must be a sequence of strings"
                )
            commands[str(name)] = [str(part) for part in cmd]
        return commands


class AppConfig(BaseModel):
    """Root application configuration grouping all sections."""

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    strategy: StrategyConfig
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    active_learning: ActiveLearningConfig | None = None
    mlflow: MLflowConfig | None = None
    alerting: AlertingConfig | None = None
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _coerce_sections(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        data = dict(values)

        def _move_section(section: str, model: type[BaseModel]) -> None:
            if section in data and isinstance(data[section], Mapping):
                return
            if section in data:
                return
            section_data: dict[str, Any] = {}
            for field in model.model_fields:
                if field in data:
                    section_data[field] = data.pop(field)
            if section_data:
                data[section] = section_data

        _move_section("training", TrainingConfig)
        _move_section("features", FeaturesConfig)
        _move_section("strategy", StrategyConfig)
        _move_section("services", ServicesConfig)
        if "model_type" in data:
            training_section = data.setdefault("training", {})
            if isinstance(training_section, Mapping):
                training_section = dict(training_section)
            training_section.setdefault("model_type", data.pop("model_type"))
            data["training"] = training_section
        return data

    def get(self, key: str, default: Any | None = None) -> Any:
        for section_name in ("training", "features", "strategy", "services"):
            section = getattr(self, section_name, None)
            if isinstance(section, BaseModel) and key in section.model_fields:
                return getattr(section, key)
        extras = getattr(self, "__pydantic_extra__", None) or {}
        return extras.get(key, default)

    def update_from(self, other: "AppConfig") -> None:
        """In-place update from another :class:`AppConfig` instance."""

        merged_data: dict[str, Any] = {
            **self.model_dump(),
            **(getattr(self, "__pydantic_extra__", None) or {}),
        }
        merged_data.update(other.model_dump())
        merged_data.update(getattr(other, "__pydantic_extra__", None) or {})
        updated = type(self).model_validate(merged_data)
        self.__dict__.update(updated.__dict__)
        object.__setattr__(
            self,
            "__pydantic_extra__",
            getattr(updated, "__pydantic_extra__", None),
        )
