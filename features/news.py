"""News and calendar related features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.events import get_events

# Default transformer used for sentiment analysis.  Tests may monkeypatch this
# to a lightweight model.
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
_MODEL_CACHE = None  # Lazily initialised (tokenizer, model)
_SENT_CACHE: dict[str, tuple[np.ndarray, float]] = {}


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

    text_col = next((c for c in ("news_summary", "summary") if c in df.columns), None)
    if text_col is None:
        df["news_sentiment"] = 0.0
        df["news_emb_0"] = 0.0
        return df

    try:  # Optional heavy dependency
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except Exception:  # pragma: no cover - optional dependency
        df["news_sentiment"] = 0.0
        df["news_emb_0"] = 0.0
        return df

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
            df["news_sentiment"] = 0.0
            df["news_emb_0"] = 0.0
            return df

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

    if emb_dim == 0:
        df["news_sentiment"] = 0.0
        df["news_emb_0"] = 0.0
        return df

    for i in range(emb_dim):
        df[f"news_emb_{i}"] = [vec[i] if len(vec) > i else 0.0 for vec in embeddings]
    df["news_sentiment"] = sentiments
    return df


def compute(df: pd.DataFrame) -> pd.DataFrame:
    from data import features as base

    df = base.add_economic_calendar_features(df)
    df = base.add_news_sentiment_features(df)
    return df


__all__ = [
    "add_economic_calendar_features",
    "add_news_sentiment_features",
    "compute",
]
