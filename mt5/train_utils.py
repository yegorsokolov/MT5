from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Tuple
from collections.abc import Sequence
import contextlib
import types

import numpy as np
import pandas as pd
import re
from mt5.log_utils import setup_logging
from utils import load_config
try:  # pragma: no cover - optional dependency
    from analytics import mlflow_client as mlflow
except Exception:  # pragma: no cover - fallback for tests
    mlflow = types.SimpleNamespace(
        start_run=lambda *a, **k: contextlib.nullcontext(),
        end_run=lambda *a, **k: None,
        log_param=lambda *a, **k: None,
        log_metric=lambda *a, **k: None,
    )
from analysis.feature_families import group_by_family
from analysis.feature_selector import select_features
from analysis.purged_cv import PurgedTimeSeriesSplit
from pandas.api.types import is_numeric_dtype

_PRICE_PATTERN = re.compile(r"price_window_(?P<step>\d+)(?:_(?P<feature>.+))?")
_NEWS_PATTERN = re.compile(r"news_emb_(?P<step>\d+)(?:_(?P<feature>\d+))?")

_DEFAULT_TARGET_COLUMNS = {"tb_label", "pseudo_label"}
_DEFAULT_ID_COLUMNS = {"Timestamp", "Symbol", "symbol", "run_id", "source", "SymbolGroup"}


def setup_training(config: Optional[str | Path | Mapping[str, Any]] = None, *, experiment: Optional[str] = None) -> Mapping[str, Any]:
    """Configure logging, load config and start an MLflow run.

    Parameters
    ----------
    config:
        Optional configuration path or mapping.  When ``None`` the default
        ``config.yaml`` is loaded.  If a mapping is supplied it is returned as
        is.
    experiment:
        Optional MLflow experiment name.  When provided, ``mlflow.start_run`` is
        invoked and the configuration is logged.

    Returns
    -------
    Mapping[str, Any]
        The resolved configuration dictionary.
    """
    setup_logging()
    if isinstance(config, (str, Path)):
        cfg_obj = load_config(config)
        cfg: Mapping[str, Any] = cfg_obj.model_dump()  # type: ignore[attr-defined]
    elif config is None:
        cfg_obj = load_config()
        cfg = cfg_obj.model_dump()  # type: ignore[attr-defined]
    else:
        cfg = dict(config)
    if experiment:
        try:
            mlflow.start_run(experiment, cfg)
        except Exception:  # pragma: no cover - mlflow optional
            pass
    return cfg


def end_training() -> None:
    """End the current MLflow run if one is active."""
    try:
        mlflow.end_run()
    except Exception:  # pragma: no cover - mlflow optional
        pass


def extract_price_windows(df: pd.DataFrame) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Return price window tensor and the contributing columns.

    The function searches ``df`` for columns prefixed with ``price_window_``.
    Columns following the pattern ``price_window_{t}`` or
    ``price_window_{t}_{feature}`` are grouped by their window index ``t`` and
    arranged into a tensor of shape ``(rows, window, features)``.  Missing
    columns result in ``None``.
    """

    matches: list[tuple[str, int, str]] = []
    for col in df.columns:
        match = _PRICE_PATTERN.match(col)
        if not match:
            continue
        step = int(match.group("step"))
        feature = match.group("feature") or "value"
        matches.append((col, step, feature))
    if not matches:
        return None
    steps = sorted({step for _, step, _ in matches})
    features = sorted({feature for _, _, feature in matches})
    step_idx = {step: idx for idx, step in enumerate(steps)}
    feat_idx = {feature: idx for idx, feature in enumerate(features)}
    tensor = np.zeros((len(df), len(steps), len(features)), dtype=np.float32)
    for col, step, feature in matches:
        tensor[:, step_idx[step], feat_idx[feature]] = df[col].to_numpy(dtype=np.float32)
    used_cols = [col for col, _, _ in sorted(matches, key=lambda item: (item[1], item[2]))]
    return tensor, used_cols


def extract_news_embeddings(df: pd.DataFrame) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Return news embedding tensor and contributing columns.

    News columns may be provided either as ``news_emb_{i}`` representing a
    single embedding vector per row or as ``news_emb_{step}_{dim}`` representing
    multiple news items with their own embedding dimensions.  The output tensor
    therefore has shape ``(rows, news_tokens, embedding_dim)``.
    """

    matches: list[tuple[str, int, Optional[int]]] = []
    for col in df.columns:
        match = _NEWS_PATTERN.match(col)
        if not match:
            continue
        step = int(match.group("step"))
        feature = match.group("feature")
        feat_idx = int(feature) if feature is not None else None
        matches.append((col, step, feat_idx))
    if not matches:
        return None
    if any(feat is not None for _, _, feat in matches):
        steps = sorted({step for _, step, _ in matches})
        feats = sorted({feat if feat is not None else -1 for _, _, feat in matches})
        step_idx = {step: idx for idx, step in enumerate(steps)}
        feat_idx = {feat: idx for idx, feat in enumerate(feats)}
        tensor = np.zeros((len(df), len(steps), len(feats)), dtype=np.float32)
        for col, step, feat in matches:
            feature_key = feat if feat is not None else -1
            tensor[:, step_idx[step], feat_idx[feature_key]] = df[col].to_numpy(
                dtype=np.float32
            )
    else:
        matches.sort(key=lambda item: item[1])
        ordered_cols = [col for col, _, _ in matches]
        tensor = df[ordered_cols].to_numpy(dtype=np.float32)[:, None, :]
    used_cols = [col for col, _, _ in sorted(matches, key=lambda item: (item[1], item[2] or -1))]
    return tensor, used_cols


def prepare_modal_arrays(
    df: pd.DataFrame,
    label: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, List[str], List[str]]]:
    """Prepare price/news tensors aligned with optional labels.

    Returns masked price and news tensors with finite values only.  The mask
    array has length ``len(df)`` and indicates which rows were retained.  When a
    label array is supplied, it is filtered using the same mask.
    """

    price_result = extract_price_windows(df)
    news_result = extract_news_embeddings(df)
    if price_result is None or news_result is None:
        return None
    price_tensor, price_cols = price_result
    news_tensor, news_cols = news_result
    mask = np.isfinite(price_tensor).all(axis=(1, 2)) & np.isfinite(news_tensor).all(axis=(1, 2))
    price_clean = price_tensor[mask]
    news_clean = news_tensor[mask]
    labels_clean: Optional[np.ndarray] = None
    if label is not None:
        label_arr = np.asarray(label, dtype=np.float32)
        labels_clean = label_arr[mask]
    return price_clean, news_clean, labels_clean, mask, price_cols, news_cols


def resolve_group_labels(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return a numeric array of group labels for ``df`` if available."""

    for column in ("SymbolGroup", "SymbolCode", "Symbol"):
        if column not in df.columns:
            continue
        series = df[column]
        if is_numeric_dtype(series):
            return series.to_numpy()
        return series.astype("category").cat.codes.to_numpy()
    return None


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Return configuration value ``key`` from ``cfg`` with fallback."""

    getter = getattr(cfg, "get", None)
    if callable(getter):
        value = getter(key, default)
        return default if value is None else value

    training = getattr(cfg, "training", None)
    if training is not None and hasattr(training, key):
        value = getattr(training, key)
        return default if value is None else value

    if isinstance(cfg, Mapping):
        training_section = cfg.get("training")
        if isinstance(training_section, Mapping) and key in training_section:
            value = training_section[key]
            return default if value is None else value
        value = cfg.get(key, default)
        return default if value is None else value

    return default


def _normalise_iterable(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [str(k) for k in value.keys()]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    return [str(value)]


def resolve_training_features(
    df: pd.DataFrame,
    target: Optional[pd.Series],
    cfg: Any,
    *,
    id_columns: Iterable[str] | None = None,
    target_columns: Iterable[str] | None = None,
    mandatory: Iterable[str] | None = None,
) -> List[str]:
    """Derive the training feature list from ``df`` respecting configuration."""

    numeric_columns = set(df.select_dtypes(include=[np.number, "bool"]).columns)

    target_cols = set(_DEFAULT_TARGET_COLUMNS)
    if target is not None and getattr(target, "name", None):
        target_cols.add(str(target.name))
    if target_columns:
        target_cols.update(str(col) for col in target_columns)

    id_cols = set(_DEFAULT_ID_COLUMNS)
    if id_columns:
        id_cols.update(str(col) for col in id_columns)

    candidates = [
        col
        for col in df.columns
        if col in numeric_columns and col not in target_cols and col not in id_cols
    ]

    includes_raw = _cfg_get(cfg, "feature_includes", [])
    excludes_raw = _cfg_get(cfg, "feature_excludes", [])
    families_raw = _cfg_get(cfg, "feature_families", {})
    use_selector = bool(_cfg_get(cfg, "use_feature_selector", True))
    selector_top_k = _cfg_get(cfg, "feature_selector_top_k", None)
    corr_threshold = _cfg_get(cfg, "feature_selector_corr_threshold", 0.95)

    include_cols = [col for col in _normalise_iterable(includes_raw) if col in df.columns]
    exclude_cols = {col for col in _normalise_iterable(excludes_raw) if col in df.columns}

    mandatory_cols = set(mandatory or [])
    mandatory_cols = {col for col in mandatory_cols if col in df.columns}

    override_map: dict[str, bool] = {}
    if isinstance(families_raw, Mapping):
        override_map = {
            str(name).lower(): bool(value)
            for name, value in families_raw.items()
            if value is not None
        }

    reference_cols = set(candidates) | set(include_cols) | mandatory_cols
    family_map = group_by_family(reference_cols)

    forced = set(col for col in include_cols if col in numeric_columns)
    forced.update(col for col in mandatory_cols if col in numeric_columns)

    for family, include in override_map.items():
        cols = family_map.get(family, set())
        if not cols:
            continue
        if include:
            forced.update(col for col in cols if col in numeric_columns)
        else:
            candidates = [
                c
                for c in candidates
                if c not in cols or c in include_cols
            ]
            forced.difference_update(col for col in cols if col not in include_cols)

    selected: List[str]
    top_k: Optional[int]
    try:
        top_k = int(selector_top_k) if selector_top_k is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        top_k = None

    try:
        corr_val = float(corr_threshold) if corr_threshold is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        corr_val = 0.95

    if use_selector and target is not None and candidates:
        selected = select_features(df[candidates], target, top_k=top_k, corr_threshold=corr_val)
    else:
        selected = list(candidates)

    final_set = {col for col in selected if col in numeric_columns}
    final_set.update(forced)

    explicit_keep = set(forced)

    for col in target_cols:
        final_set.discard(col)

    for col in id_cols:
        if col in final_set and col not in explicit_keep:
            final_set.discard(col)

    for col in exclude_cols:
        if col in final_set and col not in mandatory_cols:
            final_set.discard(col)

    final_set = {col for col in final_set if col in numeric_columns}

    if not final_set:
        fallback = [col for col in candidates if col in numeric_columns]
        if fallback:
            final_set.update(fallback)
        elif forced:
            final_set.update(forced)

    ordered: List[str] = []
    for col in df.columns:
        if col in final_set and col not in ordered:
            ordered.append(col)
    return ordered


def generate_time_series_folds(
    n_samples: int,
    *,
    n_splits: int = 1,
    test_size: Optional[int] = None,
    embargo: int = 0,
    min_train_size: Optional[int] = None,
    group_gap: int = 0,
    groups: Sequence | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return chronological train/validation folds using purged CV."""

    if n_samples <= 0:
        return []

    n_splits = max(1, int(n_splits))
    embargo = max(0, int(embargo))
    group_gap = max(0, int(group_gap))
    resolved_test = test_size if test_size is not None else n_samples // (n_splits + 1)
    if resolved_test <= 0:
        resolved_test = 1
    resolved_test = min(int(resolved_test), n_samples)
    resolved_min_train = None
    if min_train_size is not None:
        resolved_min_train = max(1, min(int(min_train_size), n_samples - 1))

    splitter = PurgedTimeSeriesSplit(
        n_splits=n_splits,
        embargo=embargo,
        test_size=resolved_test,
        min_train_size=resolved_min_train,
        group_gap=group_gap,
    )

    indices = list(range(n_samples))
    group_seq = list(groups) if groups is not None else None
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        for train_idx, val_idx in splitter.split(indices, groups=group_seq):
            folds.append((np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)))
    except ValueError:
        fallback_val = max(1, resolved_test)
        train_end = max(0, n_samples - fallback_val)
        folds = [
            (
                np.arange(train_end, dtype=int),
                np.arange(train_end, n_samples, dtype=int),
            )
        ]
    return folds


__all__ = [
    "setup_training",
    "end_training",
    "extract_price_windows",
    "extract_news_embeddings",
    "prepare_modal_arrays",
    "resolve_group_labels",
    "resolve_training_features",
    "generate_time_series_folds",
]
