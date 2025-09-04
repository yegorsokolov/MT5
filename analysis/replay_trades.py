"""Replay historical trade decisions with new model versions.

This utility loads previously recorded decision logs and reprocesses the
features through newly enabled model versions.  The resulting comparison
between the original probabilities and the reprocessed probabilities is
written to ``reports/replays`` for operator review.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from log_utils import read_decisions
from utils import load_config

REPLAY_DIR = Path(__file__).resolve().parent.parent / "reports" / "replays"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)


def replay_trades(model_versions: List[str]) -> None:
    """Generate comparison reports for the given model versions.

    Parameters
    ----------
    model_versions:
        List of model version identifiers to replay through.
    """

    if not model_versions:
        return
    decisions = read_decisions()
    if decisions.empty:
        return
    preds = decisions[decisions["event"] == "prediction"].copy()
    if preds.empty:
        return

    cfg = load_config()
    # Import inside function to avoid circular dependency during import time
    from generate_signals import load_models

    models = load_models([], model_versions)
    if not models:
        return

    exclude = {
        "timestamp",
        "event",
        "Symbol",
        "prob",
        "algorithm",
        "position_size",
        "reason",
        "issues",
    }
    feature_cols = [c for c in preds.columns if c not in exclude]

    summary_rows = []
    for vid, model in zip(model_versions, models):
        try:
            new_prob = model.predict_proba(preds[feature_cols])[:, 1]
        except Exception:  # pragma: no cover - defensive
            continue
        base_cols = ["timestamp", "Symbol", "prob"]
        for col in ["algorithm", "position_size", "reason", "issues"]:
            if col in preds.columns:
                base_cols.append(col)
        comp = preds[base_cols].copy()
        comp["new_prob"] = new_prob
        comp["abs_diff"] = (comp["prob"] - comp["new_prob"]).abs()
        comp.to_parquet(REPLAY_DIR / f"{vid}.parquet", index=False)
        summary_rows.append({"version": vid, "mae": comp["abs_diff"].mean()})

    if summary_rows:
        pd.DataFrame(summary_rows).to_parquet(REPLAY_DIR / "summary.parquet", index=False)
