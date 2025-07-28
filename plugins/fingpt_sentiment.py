from . import register_feature
import pandas as pd
import functools
import logging

try:
    from utils import load_config
except Exception:  # pragma: no cover - optional
    load_config = lambda: {}

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


@functools.lru_cache()
def _get_summary_pipeline():
    if pipeline is None:
        return None
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:  # pragma: no cover - download/initialization may fail
        logger.warning("Failed to load summarization model: %s", e)
        return None


@register_feature
def score_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add a FinGPT sentiment score for each row with a 'event' or 'text' column."""
    if df.empty:
        return df

    text_col = "text" if "text" in df.columns else "event" if "event" in df.columns else None
    if text_col is None:
        return df

    cfg = load_config()

    pipe = _get_pipeline()
    summary_pipe = _get_summary_pipeline() if cfg.get("use_fingpt_summary", False) else None

    out_df = df.copy()

    if pipe is not None:
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
        out_df["sentiment"] = scores
    else:
        out_df["sentiment"] = 0.0

    if summary_pipe is not None:
        summaries = summary_pipe(df[text_col].astype(str).tolist())
        out_df["summary"] = [s.get("summary_text", "") for s in summaries]
    else:
        out_df["summary"] = ""

    return out_df

