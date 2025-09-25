from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Literal, Optional

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

    telegram_bot_token: str | None = Field(
        None, description="Telegram bot token used for alert delivery"
    )
    telegram_chat_id: str | None = Field(
        None, description="Telegram chat identifier receiving alerts"
    )
    smtp: SMTPConfig | None = None
    model_config = ConfigDict(extra="forbid")

    @field_validator("telegram_chat_id", mode="before")
    @classmethod
    def _coerce_chat_id(cls, value: Any) -> str | None:
        if value is None or value == "":
            return None
        return str(value)


class AutoUpdateConfig(BaseModel):
    """Settings controlling the Git auto-update process."""

    enabled: bool = True
    remote: str = Field(
        "origin", description="Git remote name that should be tracked"
    )
    branch: str = Field(
        "main", description="Branch name that should be deployed on the VPS"
    )
    service_name: Optional[str] = Field(
        "mt5bot",
        description="systemd service restarted after a successful update",
    )
    restart_command: Optional[List[str]] = Field(
        default=None,
        description="Explicit command executed after pulling new code. Overrides service_name when provided.",
    )
    prefer_quiet_hours: bool = Field(
        True,
        description="Delay updates until markets are mostly closed when possible",
    )
    max_open_fraction: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of tracked symbols that may be open before deferring an update",
    )
    max_defer_minutes: float = Field(
        240.0,
        ge=0.0,
        description="Upper bound on how long an update can be deferred even if markets stay open",
    )
    check_interval_minutes: float = Field(
        15.0,
        ge=1.0,
        description="Recommended cadence for the auto-update timer",
    )
    fallback_exchange: str = Field(
        "24/5",
        description="Exchange identifier assumed when a symbol is missing from the explicit map",
    )
    exchanges: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from trading symbol to exchange calendar identifier",
    )
    protected_paths: List[str] = Field(
        default_factory=lambda: ["logs", "reports", "checkpoints", "models"],
        description="Directories preserved across updates",
    )
    state_file: Optional[str] = Field(
        None,
        description="Override path used to persist auto-update state",
    )
    lock_file: Optional[str] = Field(
        None,
        description="Optional path of the advisory auto-update lock file",
    )
    dry_run: bool = Field(
        False,
        description="When true, only log actions without applying updates",
    )
    model_config = ConfigDict(extra="forbid")

    @field_validator("restart_command", mode="before")
    @classmethod
    def _validate_restart_command(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            raise TypeError("restart_command must be provided as a sequence of strings")
        if isinstance(value, Sequence):
            return [str(v) for v in value]
        raise TypeError("restart_command must be a sequence of strings")

    @field_validator("protected_paths", mode="before")
    @classmethod
    def _normalise_paths(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(v).strip() for v in value if str(v).strip()]
        raise TypeError("protected_paths must be a list of directory names")


class TrainingConfig(BaseModel):
    """Training-related configuration."""

    seed: int = 42
    model_type: str = Field(
        "lgbm",
        description="Primary model architecture to train (lgbm, neural, cross_modal)",
    )
    batch_size: int | None = Field(
        None,
        ge=1,
        description="Mini-batch size for offline training runs",
    )
    eval_batch_size: int | None = Field(
        None,
        ge=1,
        description="Batch size used for evaluation loaders",
    )
    min_batch_size: int = Field(
        8,
        ge=1,
        description="Lower bound when auto-tuning training batch size",
    )
    online_batch_size: int | None = Field(
        1000,
        ge=1,
        description="Chunk size when replaying historical data online",
    )
    n_jobs: int | None = Field(
        None,
        ge=1,
        description="Parallel worker count for tree-based estimators",
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
    feature_includes: List[str] = Field(
        default_factory=list,
        description="Feature columns to always include in training",
    )
    feature_excludes: List[str] = Field(
        default_factory=list,
        description="Feature columns to drop from training",
    )
    feature_families: Dict[str, bool] = Field(
        default_factory=dict,
        description="Map of feature family name to inclusion flag",
    )
    feature_groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Custom named collections of feature columns",
    )
    use_feature_selector: bool = Field(
        True,
        description="Run analysis.feature_selector.select_features on candidates",
    )
    feature_selector_top_k: int | None = Field(
        None,
        ge=1,
        description="Optional top-k cutoff when using the feature selector",
    )
    feature_selector_corr_threshold: float | None = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Correlation threshold for dropping redundant features",
    )
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

    @field_validator("feature_groups", mode="before")
    @classmethod
    def _normalise_feature_groups(cls, value: Any) -> dict[str, list[str]]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("feature_groups must be provided as a mapping")
        groups: dict[str, list[str]] = {}
        for name, cols in value.items():
            if cols is None:
                groups[str(name)] = []
                continue
            if isinstance(cols, (str, bytes)):
                groups[str(name)] = [str(cols)]
            elif isinstance(cols, Sequence):
                groups[str(name)] = [str(c) for c in cols if str(c)]
            else:
                raise TypeError(
                    "feature_groups entries must be provided as strings or sequences"
                )
        return groups


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


class ExternalSourceConfig(BaseModel):
    """External data source specification used for context augmentation."""

    name: str = Field(..., description="Human-friendly identifier for the source")
    url: str = Field(..., description="Endpoint used to retrieve the dataset")
    method: Literal["GET", "POST", "PUT", "PATCH"] = Field(
        "GET", description="HTTP method used for the request"
    )
    format: Literal["json", "csv"] = Field(
        "json", description="Response format emitted by the endpoint"
    )
    enabled: bool = Field(True, description="Disable the source without removing it")
    params: Dict[str, Any] = Field(default_factory=dict, description="Query string parameters")
    headers: Dict[str, Any] = Field(default_factory=dict, description="HTTP headers")
    payload: Dict[str, Any] | None = Field(
        None, description="Request payload used for POST/PUT/PATCH requests"
    )
    timeout: float = Field(15.0, ge=1.0, description="HTTP request timeout in seconds")
    records_path: List[str] | None = Field(
        default=None,
        description="Nested keys to follow when extracting rows from a JSON response",
    )
    timestamp_key: str = Field(
        "timestamp",
        description="Column containing timestamps before renaming",
    )
    value_key: str | None = Field(
        None,
        description="Column containing the primary value to attach to the training frame",
    )
    value_name: str | None = Field(
        None,
        description="Rename applied to the value column once merged",
    )
    rename: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of raw response fields to dataframe column names",
    )


class ExternalContextConfig(BaseModel):
    """Configuration for augmenting history with external context."""

    enabled: bool = Field(True, description="Master switch for external context ingestion")
    join: Literal["left", "right", "inner", "outer"] = Field(
        "left", description="How external frames are joined to the base history"
    )
    sources: List[ExternalSourceConfig] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class AppConfig(BaseModel):
    """Root application configuration grouping all sections."""

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    strategy: StrategyConfig
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    external_context: ExternalContextConfig | None = None
    auto_update: AutoUpdateConfig = Field(default_factory=AutoUpdateConfig)
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
        _move_section("external_context", ExternalContextConfig)
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
            if isinstance(section, BaseModel):
                section_fields = type(section).model_fields
                if key in section_fields:
                    return getattr(section, key)
        extras = getattr(self, "__pydantic_extra__", None) or {}
        if key in extras:
            return extras[key]
        model_fields = type(self).model_fields
        if key in model_fields:
            return getattr(self, key, default)
        return default

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
