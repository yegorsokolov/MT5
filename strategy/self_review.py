"""Utilities for iterative strategy self-review."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Callable

try:  # optional dependency
    from analytics.metrics_store import record_metric, TS_PATH
except Exception:  # pragma: no cover - optional in tests
    record_metric = lambda *a, **k: None  # type: ignore
    TS_PATH = Path("analytics/metrics_timeseries.parquet")


def self_review_strategy(
    strategy: Dict[str, str],
    template_fn: Callable[[str, str, Any | None], Dict[str, str]],
    log_dir: Path,
    config: Any | None = None,
    metrics_path: Path | None = None,
) -> Dict[str, str]:
    """Refine a strategy via two self-review iterations.

    Parameters
    ----------
    strategy:
        Initial strategy draft.
    template_fn:
        Function generating strategy fields from goal and context.
    log_dir:
        Directory where intermediate drafts are stored.
    config:
        Optional configuration passed to ``template_fn``.
    metrics_path:
        Optional override for metric storage path.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "draft_0.json").open("w", encoding="utf-8") as f:
        json.dump(strategy, f)

    current = dict(strategy)
    for i in range(2):
        context = current.get("context", "")
        evaluate = current.get("evaluate", "")
        new_context = f"{context} Review {i + 1}: {evaluate}".strip()
        refined = template_fn(current.get("goal", ""), new_context, config)
        refined["goal"] = current.get("goal", "")
        refined["context"] = new_context
        with (log_dir / f"draft_{i + 1}.json").open("w", encoding="utf-8") as f:
            json.dump(refined, f)
        current = refined

    with (log_dir / "final.json").open("w", encoding="utf-8") as f:
        json.dump(current, f)

    try:
        initial_len = len(json.dumps(strategy))
        final_len = len(json.dumps(current))
        record_metric(
            "strategy_length_improvement",
            float(final_len - initial_len),
            path=metrics_path or TS_PATH,
        )
    except Exception:  # pragma: no cover - metric logging is best effort
        logging.getLogger(__name__).warning(
            "Failed to record strategy improvement metric", exc_info=True
        )

    return current
