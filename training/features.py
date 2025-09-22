"""Feature engineering helpers for the training pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import numpy as np

from analysis.feature_families import group_by_family
from analysis.risk_loss import RiskBudget
from training.data_loader import StreamingTrainingFrame

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
from mt5.config_models import AppConfig, TrainingConfig

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


def _ensure_frame(df: pd.DataFrame | object) -> pd.DataFrame:
    """Return ``df`` as a DataFrame materialising streaming inputs."""

    if isinstance(df, pd.DataFrame):
        return df
    materialise = getattr(df, "materialise", None)
    if callable(materialise):
        return materialise()
    raise TypeError("Expected DataFrame-like input")


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


def _streaming_numeric_summary(
    frame: StreamingTrainingFrame,
) -> tuple[list[str], np.ndarray | None, np.ndarray | None, list[str]]:
    """Return numeric column statistics and symbol order for ``frame``."""

    columns: list[str] = []
    sum_vec: np.ndarray | None = None
    sum_cross: np.ndarray | None = None
    total = 0
    symbol_order: list[str] = []
    seen_symbols: set[str] = set()

    for chunk in frame:
        numeric = chunk.select_dtypes(include="number")
        if not numeric.empty:
            if not columns:
                columns = list(numeric.columns)
                sum_vec = np.zeros(len(columns), dtype=float)
                sum_cross = np.zeros((len(columns), len(columns)), dtype=float)
            else:
                missing = [col for col in columns if col not in numeric.columns]
                if missing:
                    numeric = numeric.reindex(columns=columns, fill_value=0.0)
                else:
                    numeric = numeric.loc[:, columns]
            arr = numeric.to_numpy(dtype=float)
            total += arr.shape[0]
            if sum_vec is not None:
                sum_vec += arr.sum(axis=0)
            if sum_cross is not None:
                sum_cross += arr.T @ arr
        if "Symbol" in chunk.columns:
            symbols = chunk["Symbol"].dropna().astype(str)
            for sym in symbols:
                if sym not in seen_symbols:
                    seen_symbols.add(sym)
                    symbol_order.append(sym)

    if not columns or sum_vec is None or sum_cross is None or total == 0:
        return [], None, None, symbol_order

    mean = sum_vec / float(total)
    cov = sum_cross / float(total) - np.outer(mean, mean)
    return columns, mean, cov, symbol_order


def apply_domain_adaptation(
    df: pd.DataFrame | object, adapter_path: Path, *, regime_step: int = 500
) -> pd.DataFrame:
    """Apply the persisted :class:`DomainAdapter` to numeric columns."""

    from analysis.domain_adapter import DomainAdapter
    from analysis.regime_detection import periodic_reclassification

    if isinstance(df, StreamingTrainingFrame):
        adapter = DomainAdapter.load(adapter_path)
        num_cols, mean, cov, symbols = _streaming_numeric_summary(df)
        if num_cols and mean is not None and cov is not None:
            adapter.columns_ = list(num_cols)
            adapter.source_mean_ = mean
            adapter.source_cov_ = cov

            def _transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
                if not set(num_cols).issubset(chunk.columns):
                    return chunk
                numeric = chunk.loc[:, num_cols]
                if numeric.empty:
                    return chunk
                transformed = adapter.transform(numeric)
                result = chunk.copy()
                result.loc[:, num_cols] = transformed.to_numpy()
                return result

            df.apply_chunk(_transform_chunk)

        if symbols and "SymbolCode" not in df.columns:
            mapping = {sym: idx for idx, sym in enumerate(symbols)}

            def _attach_symbol_code(chunk: pd.DataFrame) -> pd.DataFrame:
                if "Symbol" not in chunk.columns or "SymbolCode" in chunk.columns:
                    return chunk
                result = chunk.copy()
                result["SymbolCode"] = (
                    result["Symbol"].map(mapping).fillna(-1).astype(int)
                )
                return result

            df.apply_chunk(_attach_symbol_code)

        adapter.save(adapter_path)
        df.apply_full(lambda frame: periodic_reclassification(frame, step=regime_step))
        return df

    df = _ensure_frame(df)
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


def append_risk_profile_features(df: pd.DataFrame | object, risk_profile) -> RiskBudget:
    """Inject user risk preferences as additional model features."""

    if isinstance(df, StreamingTrainingFrame):
        user_budget = RiskBudget(
            max_leverage=risk_profile.leverage_cap,
            max_drawdown=risk_profile.drawdown_limit,
        )

        values = {"risk_tolerance": risk_profile.tolerance}
        values.update(user_budget.as_features())

        def _attach(chunk: pd.DataFrame) -> pd.DataFrame:
            result = chunk.copy()
            for name, val in values.items():
                result[name] = val
            return result

        df.apply_chunk(_attach)
        return user_budget

    df = _ensure_frame(df)
    user_budget = RiskBudget(
        max_leverage=risk_profile.leverage_cap,
        max_drawdown=risk_profile.drawdown_limit,
    )
    df["risk_tolerance"] = risk_profile.tolerance
    for name, val in user_budget.as_features().items():
        df[name] = val
    return user_budget


def build_feature_candidates(
    df: pd.DataFrame | object,
    budget: RiskBudget | None = None,
    *,
    include_symbol_code: bool = True,
    cfg: AppConfig | TrainingConfig | Mapping[str, Any] | None = None,
) -> list[str]:
    """Return the baseline list of feature column names respecting config."""

    if isinstance(df, StreamingTrainingFrame):
        column_order = df.collect_columns()
        column_set = set(column_order)
    else:
        df = _ensure_frame(df)
        column_order = list(df.columns)
        column_set = set(column_order)
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
        if "risk_tolerance" in column_set:
            features.append("risk_tolerance")
        for name in budget.as_features().keys():
            if name in column_set:
                features.append(name)

    for prefix in _NUMERIC_PREFIXES:
        for col in column_order:
            if col.startswith(prefix):
                features.append(col)

    if "volume_ratio" in column_set:
        features.append("volume_ratio")
        if "volume_imbalance" in column_set:
            features.append("volume_imbalance")

    if include_symbol_code and "SymbolCode" in column_set:
        features.append("SymbolCode")

    features = _dedupe_preserve_order(features)

    explicit_includes = {col for col in includes if col in column_set}

    family_map = group_by_family(set(column_set) | set(features))

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
            ordered_cols = [col for col in column_order if col in cols]
            for col in ordered_cols:
                if col in column_set:
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
        if col in column_set and col not in features:
            features.append(col)

    features = [col for col in features if col in column_set]

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
