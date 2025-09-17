"""Feature engineering helpers for the training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from analysis.risk_loss import RiskBudget

__all__ = [
    "apply_domain_adaptation",
    "append_risk_profile_features",
    "build_feature_candidates",
    "ensure_mandatory_features",
    "select_model_features",
]


_NUMERIC_PREFIXES = ("cross_corr_", "factor_", "cross_mom_")


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
) -> list[str]:
    """Return the baseline list of feature column names."""

    features = [
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
    if budget is not None:
        features.append("risk_tolerance")
        features.extend(budget.as_features().keys())
    for prefix in _NUMERIC_PREFIXES:
        features.extend(col for col in df.columns if col.startswith(prefix))
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    if include_symbol_code and "SymbolCode" in df.columns:
        features.append("SymbolCode")
    return list(dict.fromkeys(features))


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
