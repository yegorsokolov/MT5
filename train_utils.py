from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import re

from log_utils import setup_logging
from utils import load_config
from analytics import mlflow_client as mlflow

_PRICE_PATTERN = re.compile(r"price_window_(?P<step>\d+)(?:_(?P<feature>.+))?")
_NEWS_PATTERN = re.compile(r"news_emb_(?P<step>\d+)(?:_(?P<feature>\d+))?")


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


__all__ = [
    "setup_training",
    "end_training",
    "extract_price_windows",
    "extract_news_embeddings",
    "prepare_modal_arrays",
]
