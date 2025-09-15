from __future__ import annotations

"""Logic-oriented news sentiment analysis using transformer models."""

from functools import cached_property
from typing import Dict, Sequence


class LogicSentimentAnalyzer:
    """Analyse news text using a zero-shot classification LLM.

    The analyser employs a natural language inference model to reason about
    the most plausible sentiment label for a given headline.  Using a
    zero-shot pipeline allows flexible label sets while providing a form of
    lightweight logical reasoning via textual entailment.
    """

    def __init__(self, model: str = "roberta-large-mnli") -> None:
        self.model_name = model

    @cached_property
    def _classifier(self):
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers library is required for sentiment analysis") from exc
        return pipeline("zero-shot-classification", model=self.model_name)

    def analyze(self, text: str, labels: Sequence[str] | None = None) -> Dict[str, float]:
        """Return the best sentiment label and score for ``text``.

        Parameters
        ----------
        text:
            The input string to analyse.
        labels:
            Candidate labels to classify against.  Defaults to
            ``["positive", "negative", "neutral"]``.
        """

        labels = list(labels) if labels else ["positive", "negative", "neutral"]
        result = self._classifier(text, candidate_labels=labels)
        return {"label": result["labels"][0], "score": float(result["scores"][0])}


__all__ = ["LogicSentimentAnalyzer"]
