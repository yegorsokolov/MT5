from . import register_feature
import pandas as pd
import functools
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - transformers optional in tests
    pipeline = None


@functools.lru_cache()
def _get_pipeline():
    if pipeline is None:
        return None
    try:
        return pipeline("sentiment-analysis", model="FinGPT/fingpt-sentiment_llama2-13b_lora")
    except Exception as e:  # pragma: no cover - download/initialization may fail
        logger.warning("Failed to load FinGPT model: %s", e)
        return None


@register_feature
def score_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add a FinGPT sentiment score for each row with a 'event' or 'text' column."""
    if df.empty:
        return df

    text_col = "text" if "text" in df.columns else "event" if "event" in df.columns else None
    if text_col is None:
        return df

    pipe = _get_pipeline()
    if pipe is None:
        return df.assign(sentiment=0.0)

    outputs = pipe(df[text_col].astype(str).tolist())
    scores = []
    for out in outputs:
        label = str(out.get("label", "")).lower()
        score = float(out.get("score", 0.0))
        if label == "positive":
            scores.append(score)
        elif label == "negative":
            scores.append(-score)
        else:
            scores.append(0.0)
    df = df.copy()
    df["sentiment"] = scores
    return df
