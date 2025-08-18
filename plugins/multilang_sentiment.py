"""Multilingual sentiment analysis feature plugin.

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
from typing import Optional, List

from utils.resource_monitor import monitor

logger = logging.getLogger(__name__)

# Periodically refresh hardware capabilities
monitor.start()
TIERS = {"lite": 0, "standard": 1, "gpu": 2, "hpc": 3}

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - transformers optional in tests
    pipeline = None

try:  # pragma: no cover - optional detection
    from langdetect import detect
except Exception:  # pragma: no cover - langdetect optional
    detect = None


@functools.lru_cache()
def _get_xlm_pipeline(mode: str | None):
    """Return a multilingual sentiment pipeline if possible."""
    if pipeline is None:
        return None
    if mode in (None, "auto"):
        mode = monitor.capabilities.capability_tier()
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    if mode in ("gpu", "hpc"):
        model_name = "cardiffnlp/twitter-xlm-roberta-large-sentiment"
    try:
        return pipeline("sentiment-analysis", model=model_name)
    except Exception as e:  # pragma: no cover - download/initialization may fail
        logger.warning("Failed to load XLM-R model: %s", e)
        return None


@functools.lru_cache()
def _get_en_pipeline():
    """Return an English sentiment pipeline used after translation."""
    if pipeline is None:
        return None
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    except Exception as e:  # pragma: no cover - download/initialization may fail
        logger.warning("Failed to load English sentiment model: %s", e)
        return None


@functools.lru_cache(maxsize=8)
def _get_translation_pipeline(src: str):
    if pipeline is None:
        return None
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src}-en"
        return pipeline("translation", model=model_name)
    except Exception as e:  # pragma: no cover - optional
        logger.debug("Translation model load failed for %s: %s", src, e)
        return None


@functools.lru_cache(maxsize=256)
def _translate(text: str, src: str) -> str:
    pipe = _get_translation_pipeline(src)
    if pipe is None:
        return text
    try:
        return pipe(text, max_length=400)[0]["translation_text"]
    except Exception as e:  # pragma: no cover - inference failures
        logger.debug("Translation failed for %s: %s", src, e)
        return text


def _detect_lang(text: str) -> str:
    if detect is None:
        return "en"
    try:  # pragma: no cover - heuristic
        return detect(text)
    except Exception:
        return "en"


@register_feature
def score_headlines(
    df: pd.DataFrame, mode: str = "auto", api_url: Optional[str] = None
) -> pd.DataFrame:
    """Add a multilingual sentiment score for each row with a 'headline' or 'text' column."""
    if df.empty:
        return df

    text_col = (
        "headline" if "headline" in df.columns else "text" if "text" in df.columns else None
    )
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

    pipe = _get_xlm_pipeline(mode)
    texts: List[str] = df[text_col].astype(str).tolist()
    if pipe is None:
        en_pipe = _get_en_pipeline()
        if en_pipe is None:
            return df.assign(sentiment=0.0)
        langs = (
            df["lang"].astype(str).tolist() if "lang" in df.columns else [_detect_lang(t) for t in texts]
        )
        texts = [_translate(t, src=l if l else "en") for t, l in zip(texts, langs)]
        outputs = en_pipe(texts)
    else:
        outputs = pipe(texts)

    scores: List[float] = []
    for out in outputs:
        label = str(out.get("label", "")).lower()
        score = float(out.get("score", 0.0))
        if "positive" in label or label.endswith("4") or label.endswith("5"):
            scores.append(score)
        elif "negative" in label or label.startswith("1") or label.startswith("2"):
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
            _get_xlm_pipeline.cache_clear()
            _get_en_pipeline.cache_clear()
            _get_translation_pipeline.cache_clear()
            current = tier


try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.get_event_loop()
loop.create_task(_watch())
