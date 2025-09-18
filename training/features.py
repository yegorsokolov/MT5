"""Feature engineering helpers for the training pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from analysis.feature_families import group_by_family
from analysis.risk_loss import RiskBudget

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from config_models import AppConfig, TrainingConfig

__all__ = [
    "apply_domain_adaptation",
    "append_risk_profile_features",
    "build_feature_candidates",
    "ensure_mandatory_features",
    "select_model_features",
]


_NUMERIC_PREFIXES = ("cross_corr_", "factor_", "cross_mom_")
_DEFAULT_BASE_FEATURES = [
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


def _dedupe_preserve_order(columns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def _resolve_training_section(cfg: Any | None) -> Any | None:
    """Return the training section from ``cfg`` when available."""

    if cfg is None:
        return None
    if hasattr(cfg, "feature_includes"):
        return cfg
    if hasattr(cfg, "training"):
        training = getattr(cfg, "training")
        if training is not None:
            return training
    if isinstance(cfg, Mapping):
        training_section = cfg.get("training")
        if training_section is not None:
            return training_section
    return cfg


def _get_training_option(section: Any | None, key: str, default: Any) -> Any:
    """Read ``key`` from ``section`` falling back to ``default``."""

    if section is None:
        return default
    if isinstance(section, Mapping):
        value = section.get(key, default)
    else:
        value = getattr(section, key, default)
    if value is None:
        return default
    return value


def _normalise_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(v) for v in value if str(v)]
    return [str(value)]


def apply_domain_adaptation(
    df: pd.DataFrame, adapter_path: Path, *, regime_step: int = 500
) -> pd.DataFrame:
    """Apply the persisted :class:`DomainAdapter` to numeric columns."""

    from analysis.domain_adapter import DomainAdapter
    from analysis.regime_detection import periodic_reclassification

    adapter = DomainAdapter.load(adapter_path)
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        adapter.fit_source(df[num_cols])
        df.loc[:, num_cols] = adapter.transform(df[num_cols])
    adapter.save(adapter_path)
    df = periodic_reclassification(df, step=regime_step)
    if "Symbol" in df.columns and "SymbolCode" not in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes
    return df


def append_risk_profile_features(df: pd.DataFrame, risk_profile) -> RiskBudget:
    """Inject user risk preferences as additional model features."""

    user_budget = RiskBudget(
        max_leverage=risk_profile.leverage_cap,
        max_drawdown=risk_profile.drawdown_limit,
    )
    df["risk_tolerance"] = risk_profile.tolerance
    for name, val in user_budget.as_features().items():
        df[name] = val
    return user_budget


def build_feature_candidates(
    df: pd.DataFrame,
    budget: RiskBudget | None = None,
    *,
    include_symbol_code: bool = True,
    cfg: AppConfig | TrainingConfig | Mapping[str, Any] | None = None,
) -> list[str]:
    """Return the baseline list of feature column names respecting config."""

    training_section = _resolve_training_section(cfg)
    includes = _normalise_list(_get_training_option(training_section, "feature_includes", []))
    excludes = set(
        _normalise_list(_get_training_option(training_section, "feature_excludes", []))
    )

    families_raw = _get_training_option(training_section, "feature_families", {})
    group_defs = _get_training_option(training_section, "feature_groups", {})

    features: list[str] = []
    if includes:
        features.extend(includes)
    features.extend(_DEFAULT_BASE_FEATURES)

    if budget is not None:
        if "risk_tolerance" in df.columns:
            features.append("risk_tolerance")
        for name in budget.as_features().keys():
            if name in df.columns:
                features.append(name)

    for prefix in _NUMERIC_PREFIXES:
        for col in df.columns:
            if col.startswith(prefix):
                features.append(col)

    if "volume_ratio" in df.columns:
        features.append("volume_ratio")
        if "volume_imbalance" in df.columns:
            features.append("volume_imbalance")

    if include_symbol_code and "SymbolCode" in df.columns:
        features.append("SymbolCode")

    features = _dedupe_preserve_order(features)

    explicit_includes = {col for col in includes if col in df.columns}

    family_map = group_by_family(set(df.columns) | set(features))

    if isinstance(group_defs, Mapping):
        for name, cols in group_defs.items():
            columns = _normalise_list(cols)
            if not columns:
                continue
            family_map.setdefault(str(name).lower(), set()).update(columns)

    family_overrides: dict[str, bool] = {}
    if isinstance(families_raw, Mapping):
        for name, flag in families_raw.items():
            if flag is None:
                continue
            family_overrides[str(name).lower()] = bool(flag)

    for family, include_flag in family_overrides.items():
        cols = family_map.get(family, set())
        if not cols:
            continue
        if include_flag:
            ordered_cols = [col for col in df.columns if col in cols]
            for col in ordered_cols:
                if col in df.columns:
                    features.append(col)
            features = _dedupe_preserve_order(features)
        else:
            features = [
                col
                for col in features
                if not (col in cols and col not in explicit_includes)
            ]

    features = [
        col for col in features if col not in excludes or col in explicit_includes
    ]

    for col in includes:
        if col in df.columns and col not in features:
            features.append(col)

    features = [col for col in features if col in df.columns]

    return _dedupe_preserve_order(features)


def ensure_mandatory_features(
    features: list[str], mandatory: Iterable[str] | None = None
) -> list[str]:
    """Add ``mandatory`` columns to ``features`` preserving order."""

    if not mandatory:
        return features
    ordered = list(features)
    for col in mandatory:
        if col in ordered:
            continue
        ordered.append(col)
    return ordered


def select_model_features(
    df: pd.DataFrame,
    features: Sequence[str],
    target: pd.Series | None,
    *,
    model_type: str,
    mandatory: Iterable[str] | None = None,
) -> list[str]:
    """Select the final feature set used for training."""

    from analysis.feature_selector import select_features

    features = ensure_mandatory_features(list(features), mandatory)
    if model_type != "cross_modal" and target is not None and len(features):
        selected = select_features(df[features], target)
    else:
        selected = list(dict.fromkeys(features))
    if mandatory:
        for col in mandatory:
            if col in df.columns and col not in selected:
                selected.append(col)
    return selected
