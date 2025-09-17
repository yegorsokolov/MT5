"""Utilities for grouping feature columns into semantic families.

The training scripts derive candidate feature sets dynamically from the
dataframe returned by :func:`data.features.make_features`.  Many of these
columns belong to logical families such as baseline strategy signals, order
flow metrics or cross-spectral statistics.  This module provides lightweight
pattern matching helpers so callers can retain or drop entire families via
configuration without needing to hard-code column names inside the training
loops.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, MutableMapping, Sequence, Set

# Patterns are intentionally lower-cased so matching can ignore the column
# casing produced by different feature modules.
_FamilyRule = Dict[str, Sequence[str]]


FAMILY_RULES: Dict[str, _FamilyRule] = {
    "baseline": {
        "prefixes": ("baseline_", "long_stop", "short_stop"),
        "exact": (),
        "contains": (),
    },
    "order_flow": {
        "prefixes": ("imbalance", "cvd"),
        "exact": (),
        "contains": (),
    },
    "cross_spectral": {
        "prefixes": ("coh_", "cross_spec"),
        "exact": (),
        "contains": (),
    },
    "cross_asset": {
        "prefixes": ("cross_corr_", "cross_mom_", "factor_"),
        "exact": (),
        "contains": (),
    },
    "price_window": {
        "prefixes": ("price_window_",),
        "exact": (),
        "contains": (),
    },
    "news_embedding": {
        "prefixes": ("news_emb_",),
        "exact": (),
        "contains": (),
    },
    "news": {
        "prefixes": ("news_",),
        "exact": (),
        "contains": (),
    },
    "volume": {
        "prefixes": ("volume_",),
        "exact": ("obv", "mfi"),
        "contains": (),
    },
    "microprice": {
        "prefixes": ("microprice",),
        "exact": (),
        "contains": (),
    },
    "liquidity": {
        "prefixes": ("liq_",),
        "exact": (),
        "contains": (),
    },
    "ram": {
        "prefixes": ("ram",),
        "exact": (),
        "contains": (),
    },
    "divergence": {
        "prefixes": ("div_",),
        "exact": (),
        "contains": (),
    },
    "multi_timeframe": {
        "prefixes": ("htf_",),
        "exact": (),
        "contains": (),
    },
    "supertrend": {
        "prefixes": ("supertrend",),
        "exact": (),
        "contains": (),
    },
    "keltner_squeeze": {
        "prefixes": ("squeeze",),
        "exact": (),
        "contains": (),
    },
    "adaptive_ma": {
        "prefixes": ("kama",),
        "exact": (),
        "contains": (),
    },
    "kalman_ma": {
        "prefixes": ("kma",),
        "exact": (),
        "contains": (),
    },
    "regime": {
        "prefixes": ("regime", "vae_regime"),
        "exact": (),
        "contains": (),
    },
    "garch": {
        "prefixes": ("garch_",),
        "exact": (),
        "contains": (),
    },
    "frequency": {
        "prefixes": ("spec_", "wavelet_", "hurst", "fractal"),
        "exact": (),
        "contains": (),
    },
    "dtw": {
        "prefixes": ("dtw_",),
        "exact": (),
        "contains": (),
    },
    "knowledge_graph": {
        "prefixes": ("kg_",),
        "exact": (),
        "contains": ("risk_score", "opportunity_score"),
    },
    "risk": {
        "prefixes": ("risk_",),
        "exact": ("risk_tolerance",),
        "contains": (),
    },
}


def _matches_rule(column: str, rule: _FamilyRule) -> bool:
    col = column.lower()
    for name in rule.get("exact", ()):  # type: ignore[arg-type]
        if col == name:
            return True
    for prefix in rule.get("prefixes", ()):  # type: ignore[arg-type]
        if col.startswith(prefix):
            return True
    for suffix in rule.get("suffixes", ()):  # type: ignore[arg-type]
        if col.endswith(suffix):
            return True
    for fragment in rule.get("contains", ()):  # type: ignore[arg-type]
        if fragment in col:
            return True
    return False


def families_for_column(column: str) -> Set[str]:
    """Return the set of feature families that ``column`` belongs to."""

    matches: set[str] = set()
    for family, rule in FAMILY_RULES.items():
        if _matches_rule(column, rule):
            matches.add(family)
    return matches


def group_by_family(columns: Iterable[str]) -> Dict[str, Set[str]]:
    """Return mapping of family name to the columns that belong to it."""

    grouped: MutableMapping[str, Set[str]] = defaultdict(set)
    for col in columns:
        for family in families_for_column(col):
            grouped[family].add(col)
    return dict(grouped)


__all__ = ["FAMILY_RULES", "families_for_column", "group_by_family"]

