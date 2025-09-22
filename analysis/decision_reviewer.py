from __future__ import annotations

"""LLM-based review of recorded decision rationales.

This module loads the encrypted decision log, extracts any natural language
rationales and sends them to an LLM in batches for critique.  The returned
feedback is summarised and persisted to ``reports/decision_reviews`` for later
operator review.

The LLM is expected to return a plain text response containing optional
structured lines in the following format::

    SUMMARY: overall observations
    FEATURE: suggested feature engineering change
    MODEL: suggested model architecture or training change
    MANUAL: <decision id requiring manual follow-up>
    RETRAIN: <decision id suitable for automatic retraining>

Any other lines are appended to the summary verbatim.  The ``decision id``
corresponds to the index of the reviewed decision within the DataFrame.

A simple callable ``llm`` may be provided which accepts a prompt string and
returns the LLM response.  When omitted the function will attempt to use the
``openai`` package with the ``OPENAI_API_KEY`` environment variable.
"""

from pathlib import Path
from datetime import datetime, UTC
from typing import Iterable, Callable, Dict, List
import json
import os

import pandas as pd
from mt5.log_utils import read_decisions

# Directory for generated review reports
REVIEW_DIR = Path(__file__).resolve().parent.parent / "reports" / "decision_reviews"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

LLMFunc = Callable[[str], str]


def _default_llm(prompt: str) -> str:
    """Fallback LLM implementation using the ``openai`` package."""
    try:  # pragma: no cover - optional dependency
        import openai
    except Exception as exc:  # pragma: no cover - network
        raise RuntimeError("openai package not available") from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = api_key
    model = os.getenv("REVIEW_MODEL", "gpt-3.5-turbo")
    resp = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["choices"][0]["message"]["content"]


def _query_llm(prompt: str, llm: LLMFunc | None = None) -> str:
    func = llm or _default_llm
    return func(prompt)


def _parse_response(text: str) -> Dict[str, List[str] | str]:
    summary_lines: List[str] = []
    features: List[str] = []
    models: List[str] = []
    manual: List[str] = []
    retrain: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("FEATURE:"):
            features.append(line.split(":", 1)[1].strip())
        elif line.startswith("MODEL:"):
            models.append(line.split(":", 1)[1].strip())
        elif line.startswith("MANUAL:"):
            manual.append(line.split(":", 1)[1].strip())
        elif line.startswith("RETRAIN:"):
            retrain.append(line.split(":", 1)[1].strip())
        elif line.startswith("SUMMARY:"):
            summary_lines.append(line.split(":", 1)[1].strip())
        else:
            summary_lines.append(line)
    return {
        "summary": " ".join(summary_lines),
        "feature_changes": features,
        "model_changes": models,
        "manual": manual,
        "retrain": retrain,
    }


def review_rationales(
    llm: LLMFunc | None = None,
    batch_size: int = 20,
    decisions: pd.DataFrame | None = None,
) -> Dict[str, List[str]]:
    """Review decision rationales using an LLM and persist summaries.

    Parameters
    ----------
    llm:
        Callable that accepts a prompt string and returns the LLM response.
    batch_size:
        Number of rationales to include per LLM call.
    decisions:
        Optional pre-loaded decisions DataFrame.  When ``None`` the encrypted
        decision log is loaded via :func:`log_utils.read_decisions`.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of flagged decision identifiers for ``manual`` review and
        ``retrain`` suggestions.
    """

    df = decisions if decisions is not None else read_decisions()
    if df.empty or "reason" not in df.columns:
        return {"manual": [], "retrain": []}
    rationales = df.dropna(subset=["reason"]).reset_index(drop=True)
    if rationales.empty:
        return {"manual": [], "retrain": []}

    summaries: List[str] = []
    feature_changes: List[str] = []
    model_changes: List[str] = []
    manual_flags: List[str] = []
    retrain_flags: List[str] = []

    for start in range(0, len(rationales), batch_size):
        batch = rationales.iloc[start : start + batch_size]
        prompt_lines = [f"ID {idx}: {reason}" for idx, reason in batch["reason"].items()]
        prompt = (
            "Critique the following decision rationales and suggest improvements.\n"
            "Respond using lines starting with SUMMARY, FEATURE, MODEL, MANUAL or RETRAIN as needed.\n\n"
            + "\n".join(prompt_lines)
        )
        response = _query_llm(prompt, llm)
        parsed = _parse_response(response)
        if parsed["summary"]:
            summaries.append(parsed["summary"])  # type: ignore[arg-type]
        feature_changes.extend(parsed["feature_changes"])  # type: ignore[arg-type]
        model_changes.extend(parsed["model_changes"])  # type: ignore[arg-type]
        manual_flags.extend(parsed["manual"])  # type: ignore[arg-type]
        retrain_flags.extend(parsed["retrain"])  # type: ignore[arg-type]

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "summary": "\n".join(summaries),
        "feature_changes": feature_changes,
        "model_changes": model_changes,
        "flagged": {"manual": manual_flags, "retrain": retrain_flags},
    }
    fname = REVIEW_DIR / f"{datetime.now(tz=UTC):%Y%m%d_%H%M%S}.json"
    with open(fname, "w") as fh:
        json.dump(report, fh, indent=2)

    return {"manual": manual_flags, "retrain": retrain_flags}


__all__ = ["review_rationales", "REVIEW_DIR"]
