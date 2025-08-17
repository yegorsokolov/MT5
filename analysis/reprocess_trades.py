"""Reprocess recorded trades with the current model set."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import joblib
from .causal_impact import estimate_causal_impact

TRADE_HISTORY = Path("logs/trade_history.parquet")
REPORT_DIR = Path("reports/reprocess")
MODELS_DIR = Path("models")

logger = logging.getLogger(__name__)


def reprocess_trades(
    history_path: Path = TRADE_HISTORY,
    models_dir: Path = MODELS_DIR,
    report_dir: Path = REPORT_DIR,
) -> None:
    """Replay historic trades through all models and store audit reports."""
    if not history_path.exists():
        logger.info("No trade history found at %s", history_path)
        return
    try:
        df = pd.read_parquet(history_path)
    except Exception:
        logger.exception("Failed loading trade history from %s", history_path)
        return

    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "Timestamp",
            "Symbol",
            "side",
            "volume",
            "price",
            "order_id",
            "prob",
            "timestamp",
            "symbol",
        }
    ]
    if not feature_cols:
        logger.info("No feature columns available for trade reprocessing")
        return

    report_dir.mkdir(parents=True, exist_ok=True)
    for model_file in models_dir.glob("*.joblib"):
        try:
            model = joblib.load(model_file)
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(df[feature_cols])[:, 1]
            else:  # pragma: no cover - fallback
                preds = model.predict(df[feature_cols])
            out = df[["order_id", "Timestamp", "Symbol", "side"]].copy()
            if "prob" in df.columns:
                out["old_prob"] = df["prob"]
            out["new_prob"] = preds
            out["new_side"] = ["BUY" if p > 0.5 else "SELL" for p in preds]
            out_file = report_dir / f"{model_file.stem}_{pd.Timestamp.utcnow():%Y-%m-%d}.parquet"
            out.to_parquet(out_file)
            logger.info("Wrote reprocess report %s", out_file)
        except Exception:
            logger.exception("Failed processing model %s", model_file)

    try:
        estimate_causal_impact(df)
    except Exception:  # pragma: no cover - analysis optional
        logger.exception("Causal impact analysis failed")


def main() -> None:  # pragma: no cover - CLI entry point
    reprocess_trades()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
