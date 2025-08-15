"""FinBERT sentiment analysis feature plugin.

min_cpus: 2
min_mem_gb: 4
requires_gpu: false
"""

MIN_CPUS = 2
MIN_MEM_GB = 4.0
REQUIRES_GPU = False

from . import register_feature
import pandas as pd
import functools
import logging
import asyncio
from typing import Optional

from utils.resource_monitor import monitor

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
            else "ProsusAI/finbert"
        )
        return pipeline("sentiment-analysis", model=model_name)
    except Exception as e:  # pragma: no cover - download/initialization may fail
        logger.warning("Failed to load FinBERT model: %s", e)
        return None



@register_feature
def score_events(
    df: pd.DataFrame, mode: str = "auto", api_url: Optional[str] = None
) -> pd.DataFrame:
    """Add a FinBERT sentiment score for each row with a 'event' or 'text' column."""
    if df.empty:
        return df

    text_col = "text" if "text" in df.columns else "event" if "event" in df.columns else None
    if text_col is None:
        return df

    if mode == "remote" and api_url:
        try:  # pragma: no cover - network dependent
            import requests

            resp = requests.post(
                api_url, json={"texts": df[text_col].astype(str).tolist()}, timeout=10
            )
            resp.raise_for_status()
            scores = [float(s) for s in resp.json().get("scores", [])]
            if len(scores) != len(df):
                raise ValueError("Score length mismatch")
            out_df = df.copy()
            out_df["sentiment"] = scores
            return out_df
        except Exception as e:  # pragma: no cover - remote failures
            logger.warning("Remote sentiment request failed: %s", e)
            return df.assign(sentiment=0.0)

    pipe = _get_pipeline(mode)
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
    out_df = df.copy()
    out_df["sentiment"] = scores
    return out_df


async def _watch() -> None:
    q = monitor.subscribe()
    current = monitor.capability_tier
    while True:
        tier = await q.get()
        if TIERS.get(tier, 0) > TIERS.get(current, 0):
            _get_pipeline.cache_clear()
            current = tier


try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.get_event_loop()
loop.create_task(_watch())
