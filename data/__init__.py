"""Public dataset API exposing events, history and feature utilities."""

from __future__ import annotations

import importlib
from typing import Any

from .events import get_events
from .fundamental_features import load_fundamentals
from .graph_builder import build_correlation_graph, build_rolling_adjacency
from .history import (
    load_history_from_urls,
    load_history_mt5,
    load_history_config,
    load_history,
    load_history_parquet,
    load_history_memmap,
    save_history_parquet,
    load_multiple_histories,
)
from .multitimeframe import aggregate_timeframes
from .onchain_features import load_onchain_data
from .options_features import load_options_data
from .versioning import compute_hash

__all__ = [
    "get_events",
    "load_history_from_urls",
    "load_history_mt5",
    "load_history_config",
    "load_history",
    "load_history_parquet",
    "load_history_memmap",
    "save_history_parquet",
    "load_multiple_histories",
    "compute_hash",
    "add_index_features",
    "add_economic_calendar_features",
    "add_news_sentiment_features",
    "aggregate_timeframes",
    "build_correlation_graph",
    "build_rolling_adjacency",
    "load_fundamentals",
    "load_options_data",
    "load_onchain_data",
    "make_features",
    "make_features_memmap",
    "compute_rsi",
    "ma_cross_signal",
    "train_test_split",
    "make_sequence_arrays",
]

_FEATURE_EXPORTS: dict[str, tuple[str, str]] = {
    "add_index_features": (".features", "add_index_features"),
    "add_economic_calendar_features": (
        ".features",
        "add_economic_calendar_features",
    ),
    "add_news_sentiment_features": (
        ".features",
        "add_news_sentiment_features",
    ),
    "make_features": (".features", "make_features"),
    "make_features_memmap": (".features", "make_features_memmap"),
    "compute_rsi": (".features", "compute_rsi"),
    "ma_cross_signal": (".features", "ma_cross_signal"),
    "train_test_split": (".features", "train_test_split"),
    "make_sequence_arrays": (".features", "make_sequence_arrays"),
}


def __getattr__(name: str) -> Any:
    if name in _FEATURE_EXPORTS:
        module_name, attr_name = _FEATURE_EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'data' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
