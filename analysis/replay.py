"""Replay historical decisions through the current model."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from log_utils import DECISION_LOG
from generate_signals import load_models
from utils import load_config
try:  # optional if backend not configured
    from core import state_sync
except Exception:  # pragma: no cover - optional dependency
    state_sync = None

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def main() -> None:
    if state_sync:
        state_sync.pull_decisions()
    if not DECISION_LOG.exists():
        print("No decisions log found")
        return
    decisions = pd.read_parquet(DECISION_LOG)
    preds = decisions[decisions["event"] == "prediction"].copy()
    if preds.empty:
        print("No predictions to replay")
        return
    cfg = load_config()
    models = load_models(cfg.get("models", []))
    if not models:
        print("No model available for replay")
        return
    model = models[0]
    feature_cols = [c for c in preds.columns if c not in {"timestamp", "event", "Symbol", "prob"}]
    new_probs = model.predict_proba(preds[feature_cols])[:, 1]
    preds["reprocessed_prob"] = new_probs
    preds.to_parquet(REPORT_DIR / "reprocessed.parquet", index=False)
    mae = (preds["prob"] - preds["reprocessed_prob"]).abs().mean()
    report = pd.DataFrame({"mae": [mae]})
    report.to_parquet(REPORT_DIR / "summary.parquet", index=False)
    print(f"Saved reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
