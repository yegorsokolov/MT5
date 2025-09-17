"""FinGPT sentiment analysis feature plugin.

min_cpus: 4
min_mem_gb: 8
requires_gpu: true
"""

MIN_CPUS = 4
MIN_MEM_GB = 8.0
REQUIRES_GPU = True

from . import register_feature
import pandas as pd
import functools
import logging
import asyncio
from typing import Optional

from utils.resource_monitor import monitor

try:
    from utils import load_config
except Exception:  # pragma: no cover - optional
    load_config = lambda: {}

logger = logging.getLogger(__name__)

# Periodically refresh hardware capabilities
monitor.start()
TIERS = {"lite": 0, "standard": 1, "gpu": 2, "hpc": 3}

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - transformers optional in tests
    pipeline = None


@functools.lru_cache()
def _get_pipeline(mode: str | None):
    """Return a sentiment pipeline based on the requested mode."""
    if pipeline is None:
        return None
    if mode in (None, "auto"):
        mode = monitor.capabilities.capability_tier()
    try:
        model_name = (
            "distilbert-base-uncased-finetuned-sst-2-english"
            if mode in ("lite", "standard")
            else "FinGPT/fingpt-sentiment_llama2-13b_lora"
        )
        return pipeline("sentiment-analysis", model=model_name)
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


async def _watch() -> None:
    q = monitor.subscribe()
    current = monitor.capability_tier
    while True:
        tier = await q.get()
        if TIERS.get(tier, 0) > TIERS.get(current, 0):
            _get_pipeline.cache_clear()
            _get_summary_pipeline.cache_clear()
            current = tier


try:
    monitor.create_task(_watch())
except Exception:
    pass


@register_feature
def score_events(
    df: pd.DataFrame, mode: str = "auto", api_url: Optional[str] = None
) -> pd.DataFrame:
    """Add a FinGPT sentiment score for each row with a 'event' or 'text' column."""
    if df.empty:
        return df

    text_col = "text" if "text" in df.columns else "event" if "event" in df.columns else None
    if text_col is None:
        return df

    cfg = load_config()

    if mode == "remote" and api_url:
        try:  # pragma: no cover - network dependent
            import requests

            resp = requests.post(
                api_url, json={"texts": df[text_col].astype(str).tolist()}, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            scores = [float(s) for s in data.get("scores", [])]
            summaries = data.get("summaries", [])
            if len(scores) != len(df):
                raise ValueError("Score length mismatch")
            out_df = df.copy()
            out_df["sentiment"] = scores
            if summaries and len(summaries) == len(df):
                out_df["summary"] = summaries
            else:
                out_df["summary"] = ""
            return out_df
        except Exception as e:  # pragma: no cover - remote failures
            logger.warning("Remote sentiment request failed: %s", e)
            out_df = df.copy()
            out_df["sentiment"] = 0.0
            out_df["summary"] = ""
            return out_df

    pipe = _get_pipeline(mode)
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

