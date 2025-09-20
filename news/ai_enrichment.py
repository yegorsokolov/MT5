"""Lightweight AI helpers for enriching news events.

The helpers in this module intentionally avoid heavy dependencies at import
time.  When optional libraries such as :mod:`transformers` are available they
are used to provide more capable natural language processing.  Otherwise the
functions fall back to inexpensive heuristics so the news aggregation pipeline
remains functional in constrained environments (e.g. unit tests or
air-gapped deployments).

The API exposed by this module is deliberately small:

``summarize_text``
    Produce a concise summary for a news headline/paragraph.

``predict_sentiment``
    Return a sentiment score in the ``[-1, 1]`` range.

``extract_keywords``
    Extract key tokens that characterise the article.

``classify_topics``
    Map articles to broad topical buckets for downstream routing.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re
from typing import Iterable, List, Optional


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z]{3,}")

_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "have",
    "will",
    "into",
    "their",
    "after",
    "over",
    "amid",
    "amidst",
    "company",
    "companies",
    "month",
    "year",
    "years",
    "today",
    "week",
    "ahead",
    "firm",
    "firms",
}

_TOPIC_KEYWORDS = {
    "earnings": {"earnings", "profit", "profits", "forecast", "forecasts", "beats", "beat", "eps", "revenue"},
    "markets": {"stocks", "shares", "surge", "rally", "selloff", "markets", "equities", "index", "indexes", "futures"},
    "economy": {"inflation", "gdp", "growth", "employment", "jobs", "economy", "output", "recession"},
    "policy": {"fed", "federal", "bank", "rates", "policy", "interest", "reserve", "hike", "cut"},
    "crypto": {"bitcoin", "crypto", "ethereum", "blockchain", "token"},
}

_POSITIVE_TERMS = {
    "beat",
    "beats",
    "surge",
    "surges",
    "rally",
    "rallies",
    "gain",
    "gains",
    "up",
    "record",
    "growth",
    "strong",
    "optimism",
    "soar",
    "soars",
    "improve",
    "improves",
    "bullish",
    "expand",
    "expands",
}

_NEGATIVE_TERMS = {
    "fall",
    "falls",
    "drop",
    "drops",
    "down",
    "slump",
    "loss",
    "losses",
    "warning",
    "weak",
    "decline",
    "declines",
    "plunge",
    "plunges",
    "miss",
    "misses",
    "bearish",
    "shrink",
    "shrinks",
}


def _clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


@lru_cache(maxsize=2)
def _load_pipeline(task: str):  # pragma: no cover - optional dependency
    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        return None
    try:
        if task == "summarization":
            return pipeline(task, model="sshleifer/distilbart-cnn-12-6")
        if task == "sentiment-analysis":
            return pipeline(task)
        return pipeline(task)
    except Exception:
        return None


def _heuristic_summary(text: str, max_length: int = 120) -> str:
    sentences = _SENTENCE_RE.split(text)
    if not sentences:
        return text[:max_length]
    summary = sentences[0]
    if len(summary) > max_length:
        summary = summary[: max_length - 1].rstrip() + "…"
    return summary


def summarize_text(text: str, max_length: int = 120) -> Optional[str]:
    """Return a concise summary for ``text``.

    When the ``transformers`` package is installed the function will use a
    pretrained summarisation pipeline; otherwise it falls back to extracting
    the first sentence of the article.
    """

    if not text:
        return None
    cleaned = _clean_text(text)
    if not cleaned:
        return None

    summarizer = _load_pipeline("summarization")
    if summarizer is not None:  # pragma: no cover - requires heavy dependency
        try:
            result = summarizer(cleaned, max_length=max_length, min_length=max(16, max_length // 3), do_sample=False)
        except Exception:
            result = None
        if result:
            summary_text = _clean_text(result[0]["summary_text"])
            if summary_text:
                return summary_text
    return _heuristic_summary(cleaned, max_length=max_length)


def _tokenise(text: str) -> Iterable[str]:
    for match in _WORD_RE.finditer(text.lower()):
        token = match.group(0)
        if token in _STOPWORDS:
            continue
        yield token


def extract_keywords(text: str, top_n: int = 6) -> List[str]:
    """Return ``top_n`` keywords ordered by frequency."""

    if not text:
        return []
    tokens = list(_tokenise(text))
    if not tokens:
        return []
    counts = Counter(tokens)
    keywords = [word for word, _ in counts.most_common(top_n)]
    return keywords


def classify_topics(text: str) -> List[str]:
    """Classify ``text`` into coarse topics using keyword matching."""

    tokens = set(_tokenise(text))
    if not tokens:
        return []
    topics: List[str] = []
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if tokens & keywords:
            topics.append(topic)
    return topics


def _heuristic_sentiment(tokens: Iterable[str]) -> float:
    pos = 0
    neg = 0
    for token in tokens:
        if token in _POSITIVE_TERMS:
            pos += 1
        elif token in _NEGATIVE_TERMS:
            neg += 1
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / float(pos + neg)
    return max(-1.0, min(1.0, score))


def predict_sentiment(text: str) -> Optional[float]:
    """Return a sentiment score in the ``[-1, 1]`` range."""

    if not text:
        return None
    cleaned = _clean_text(text)
    if not cleaned:
        return None

    analyser = _load_pipeline("sentiment-analysis")
    if analyser is not None:  # pragma: no cover - optional dependency
        try:
            result = analyser(cleaned, truncation=True)
        except Exception:
            result = None
        if result:
            res = result[0]
            label = res.get("label", "").lower()
            score = float(res.get("score", 0.0))
            if "neg" in label:
                score = -score
            return max(-1.0, min(1.0, score * 2 - 1))

    tokens = list(_tokenise(cleaned))
    return _heuristic_sentiment(tokens)


__all__ = [
    "summarize_text",
    "predict_sentiment",
    "extract_keywords",
    "classify_topics",
]

