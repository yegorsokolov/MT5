"""Public dataset API exposing events, history and feature utilities."""

from .events import get_events
from .history import (
    load_history_from_urls,
    load_history_mt5,
    load_history_config,
    load_history,
    load_history_parquet,
    save_history_parquet,
    load_multiple_histories,
)
from .versioning import compute_hash
from .features import (
    add_index_features,
    add_economic_calendar_features,
    add_news_sentiment_features,
    make_features,
    compute_rsi,
    ma_cross_signal,
    train_test_split,
    make_sequence_arrays,
)
from .multitimeframe import aggregate_timeframes
from .graph_builder import build_correlation_graph

__all__ = [
    "get_events",
    "load_history_from_urls",
    "load_history_mt5",
    "load_history_config",
    "load_history",
    "load_history_parquet",
    "save_history_parquet",
    "load_multiple_histories",
    "compute_hash",
    "add_index_features",
    "add_economic_calendar_features",
    "add_news_sentiment_features",
    "aggregate_timeframes",
    "build_correlation_graph",
    "make_features",
    "compute_rsi",
    "ma_cross_signal",
    "train_test_split",
    "make_sequence_arrays",
]
