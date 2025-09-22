"""News and calendar related features."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import importlib.util
from types import SimpleNamespace

from data.events import get_events

try:  # pragma: no cover - decorator optional when imported standalone
    from . import validate_module
except Exception:  # pragma: no cover - fallback without validation
    def validate_module(func):
        return func

try:  # pragma: no cover - validators optional in some environments
    from .validators import require_columns, assert_no_nan
except Exception:  # pragma: no cover - graceful fallback when validators missing
    def require_columns(df, cols, **_):  # type: ignore[unused-arg]
        return df

    def assert_no_nan(df, cols=None, **_):  # type: ignore[unused-arg]
        return df

# Default transformer used for sentiment analysis.  Tests may monkeypatch this
# to a lightweight model.
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
_MODEL_CACHE = None  # Lazily initialised (tokenizer, model)
_SENT_CACHE: dict[str, tuple[np.ndarray, float]] = {}

DEFAULT_EMBED_DIM = 768


def _load_effect_length():
    try:
        from news.effect_length import estimate_effect_length  # type: ignore

        return estimate_effect_length
    except Exception:
        effect_path = Path(__file__).resolve().parents[1] / "news" / "effect_length.py"
        spec = importlib.util.spec_from_file_location("news_effect_length", effect_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[arg-type]
            return getattr(module, "estimate_effect_length", None)
    return None


_EFFECT_ESTIMATOR = _load_effect_length()


def _estimate_effect_length(
    text: str,
    magnitude: float,
    *,
    importance: float | None = None,
    risk_scale: float = 1.0,
):
    if _EFFECT_ESTIMATOR is not None:
        try:
            return _EFFECT_ESTIMATOR(
                text=text,
                magnitude=magnitude,
                importance=importance,
                risk_scale=risk_scale,
            )
        except Exception:
            pass

    try:
        mag = abs(float(magnitude))
    except (TypeError, ValueError):
        mag = 0.0
    mag = max(0.0, min(1.0, mag))
    try:
        imp = float(importance) if importance is not None else 0.3
    except (TypeError, ValueError):
        imp = 0.3
    imp = max(0.0, min(1.0, imp))
    minutes = 45.0 + 90.0 * imp + 150.0 * mag
    half_life = minutes * (0.35 + 0.45 * imp + 0.2 * mag)
    score = max(0.0, min(1.0, math.tanh(minutes / 180.0)))
    return SimpleNamespace(minutes=minutes, half_life=half_life, score=score, importance=imp)


def _bounded_tanh(value: float, scale: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(math.tanh(value / max(scale, 1e-6)))


def _market_adjustment() -> Tuple[float, float]:
    """Return market adjustment factor and intensity in ``[0, 1]``."""

    try:  # pragma: no cover - optional dependency
from mt5.market_condition import MarketConditionAssessor  # type: ignore

        assessor = MarketConditionAssessor()
        state = assessor.assess()
    except Exception:  # pragma: no cover - degraded environment
        return 1.0, 0.0

    volatility = _bounded_tanh(abs(float(state.get("volatility", 0.0))), 2.0)
    trend = _bounded_tanh(abs(float(state.get("trend_strength", 0.0))), 1.0)
    regime = _bounded_tanh(abs(float(state.get("regime", 0.0))), 2.5)
    intensity = 0.6 * volatility + 0.3 * trend + 0.1 * regime
    factor = 0.75 + 0.5 * intensity
    return float(max(0.5, min(1.5, factor))), float(max(0.0, min(1.0, intensity)))


def _risk_adjustment() -> Tuple[float, float]:
    """Return risk scaling factor and risk tolerance ratio."""

    try:  # pragma: no cover - optional persistence dependency
from mt5.state_manager import load_user_risk  # type: ignore

        cfg = load_user_risk()
    except Exception:
        return 1.0, 0.5

    try:
        daily = float(cfg.get("daily_drawdown", 0.0))
        total = float(cfg.get("total_drawdown", 0.0))
        blackout = float(cfg.get("news_blackout_minutes", 0.0))
    except Exception:
        return 1.0, 0.5

    if not math.isfinite(daily) or daily < 0:
        daily = 0.0
    if not math.isfinite(total) or total <= 0:
        total = max(daily, 1.0)

    ratio = float(max(0.05, min(1.0, daily / total if total else 0.5)))
    blackout_factor = 1.0 - min(max(blackout, 0.0) / 720.0, 0.5)
    factor = (0.75 + 0.5 * ratio) * blackout_factor
    return float(max(0.3, min(1.7, factor))), ratio


def _ensure_additional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Populate derived sentiment columns with neutral defaults if missing."""

    defaults = {
        "news_sentiment": 0.0,
        "news_sentiment_magnitude": 0.0,
        "news_sentiment_length": 0.0,
        "news_sentiment_effect_minutes": 0.0,
        "news_sentiment_effect_half_life": 0.0,
        "news_sentiment_importance": 0.0,
        "news_sentiment_market_adjusted": 0.0,
        "news_sentiment_risk_weighted": 0.0,
        "news_sentiment_severity": 0.0,
        "news_sentiment_effect": 0.0,
        "news_market_intensity": 0.0,
        "news_risk_tolerance": 0.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


def _add_zero_embeddings(df: pd.DataFrame, dim: int = DEFAULT_EMBED_DIM) -> pd.DataFrame:
    cols = [f"news_emb_{i}" for i in range(dim)]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        zeros = np.zeros((len(df), len(missing)), dtype=np.float32)
        df = df.join(pd.DataFrame(zeros, columns=missing, index=df.index))
    df.attrs["news_embedding_dim"] = dim
    return df


def add_economic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        events = get_events(past_events=True)
    except Exception:  # pragma: no cover - network issues
        events = []
    if not events:
        df["minutes_to_event"] = np.nan
        df["minutes_from_event"] = np.nan
        df["nearest_news_minutes"] = np.nan
        df["upcoming_red_news"] = 0
        return df
    event_time = pd.to_datetime(events[0]["date"]).tz_localize(None)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
    df["minutes_to_event"] = (event_time - df["Timestamp"]).dt.total_seconds() / 60
    df["minutes_from_event"] = (df["Timestamp"] - event_time).dt.total_seconds() / 60
    df["nearest_news_minutes"] = np.minimum(
        df["minutes_to_event"].abs(), df["minutes_from_event"].abs()
    )
    df["upcoming_red_news"] = (
        (df["minutes_to_event"] >= 0) & (df["minutes_to_event"] <= 60)
    ).astype(int)
    return df


def add_news_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append transformer based news sentiment features.

    The function attempts to load a HuggingFace sentiment classification model
    using :class:`~transformers.AutoModelForSequenceClassification`.  If the
    ``transformers`` (and ``torch``) libraries or the model weights are not
    available the features gracefully degrade to zeros so downstream code can
    rely on their presence.

    For each row the ``news_summary`` (or ``summary``) text is encoded and the
    pooled embedding from the final hidden state is appended as
    ``news_emb_{i}``.  A scalar polarity score ``news_sentiment`` is computed as
    ``P(positive) - P(negative)`` from the classifier logits.  Results are
    cached in-memory keyed by the raw text so repeated summaries avoid redundant
    model calls.
    """

    df = df.copy()
    text_col = next((c for c in ("news_summary", "summary") if c in df.columns), None)
    if text_col is None:
        df = _ensure_additional_columns(df)
        return _add_zero_embeddings(df)

    try:  # Optional heavy dependency
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except Exception:  # pragma: no cover - optional dependency
        df = _ensure_additional_columns(df)
        return _add_zero_embeddings(df)

    global _MODEL_CACHE, _SENT_CACHE
    try:
        tokenizer, model = _MODEL_CACHE  # type: ignore[misc]
    except Exception:
        _MODEL_CACHE = None
        _SENT_CACHE = {}
        tokenizer = model = None

    if _MODEL_CACHE is None:
        try:
            model_name = globals().get("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _MODEL_CACHE = (tokenizer, model)
        except Exception:  # pragma: no cover - model download failure
            df = _ensure_additional_columns(df)
            return _add_zero_embeddings(df)

    emb_dim = int(getattr(model.config, "hidden_size", 0))
    texts = df[text_col].fillna("").astype(str)
    embeddings: list[np.ndarray] = []
    sentiments: list[float] = []
    for text in texts:
        if text in _SENT_CACHE:
            emb, pol = _SENT_CACHE[text]
        else:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pol = float(probs[1] - probs[0])
                emb = outputs.hidden_states[-1][0, 0].detach().cpu().numpy().astype(float)
            except Exception:  # pragma: no cover - runtime error
                emb = np.zeros(emb_dim, dtype=float)
                pol = 0.0
            _SENT_CACHE[text] = (emb, pol)
        embeddings.append(emb)
        sentiments.append(pol)

    sentiments_arr = np.asarray(sentiments, dtype=np.float32)
    market_factor, market_intensity = _market_adjustment()
    risk_factor, risk_ratio = _risk_adjustment()
    magnitude = np.abs(sentiments_arr)
    effects = [
        _estimate_effect_length(text, magnitude=mag, risk_scale=risk_factor)
        for text, mag in zip(texts.tolist(), magnitude.tolist())
    ]
    length_scores = np.array([eff.score for eff in effects], dtype=np.float32)
    effect_minutes = np.array([eff.minutes for eff in effects], dtype=np.float32)
    half_life = np.array([eff.half_life for eff in effects], dtype=np.float32)
    importance_scores = np.array([eff.importance for eff in effects], dtype=np.float32)
    severity = np.clip(
        magnitude
        * (
            0.45
            + 0.3 * length_scores
            + 0.15 * float(market_intensity)
            + 0.1 * importance_scores
        )
        * float(risk_factor),
        0.0,
        1.0,
    ).astype(np.float32)
    df["news_sentiment"] = sentiments_arr
    df["news_sentiment_magnitude"] = magnitude
    df["news_sentiment_length"] = length_scores
    df["news_sentiment_effect_minutes"] = effect_minutes
    df["news_sentiment_effect_half_life"] = half_life
    df["news_sentiment_importance"] = importance_scores
    df["news_sentiment_market_adjusted"] = sentiments_arr * float(market_factor)
    df["news_sentiment_risk_weighted"] = sentiments_arr * float(risk_factor)
    df["news_sentiment_severity"] = severity
    df["news_sentiment_effect"] = sentiments_arr * severity
    df["news_market_intensity"] = market_intensity
    df["news_risk_tolerance"] = risk_ratio

    if emb_dim == 0:
        df = _add_zero_embeddings(df)
        return df

    emb_matrix = np.zeros((len(embeddings), emb_dim), dtype=np.float32)
    for row, vec in enumerate(embeddings):
        if len(vec) == 0:
            continue
        length = min(len(vec), emb_dim)
        emb_matrix[row, :length] = np.asarray(vec[:length], dtype=np.float32)

    for i in range(emb_dim):
        df[f"news_emb_{i}"] = emb_matrix[:, i]
    df.attrs["news_embedding_dim"] = emb_dim
    return df


@validate_module
def compute(df: pd.DataFrame) -> pd.DataFrame:
    from data import features as base

    require_columns(df, ["Timestamp"])
    assert_no_nan(df, ["Timestamp"])

    df = base.add_economic_calendar_features(df)
    df = base.add_news_sentiment_features(df)
    return df


__all__ = [
    "add_economic_calendar_features",
    "add_news_sentiment_features",
    "compute",
]
