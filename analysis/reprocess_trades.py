"""Reprocess recorded trades with the current model set."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import joblib
from .causal_impact import estimate_causal_impact
from analytics.trade_analyzer import TradeAnalyzer

TRADE_HISTORY = Path("logs/trade_history.parquet")
REPORT_DIR = Path("reports/reprocess")
MODELS_DIR = Path("models")
HOLD_REPORT_DIR = Path("reports/hold_duration")

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
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
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
            if "entry_time" in df.columns:
                out["entry_time"] = df["entry_time"]
            if "exit_time" in df.columns:
                out["exit_time"] = df["exit_time"]
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

    # compute PnL by holding period if exit data is available
    required = {
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "volume",
        "side",
    }
    if required.issubset(df.columns):
        try:
            trades = df[list(required)].copy()
            trades["entry_time"] = pd.to_datetime(trades["entry_time"])
            trades["exit_time"] = pd.to_datetime(trades["exit_time"])
            direction = trades["side"].str.upper().map({"BUY": 1, "SELL": -1}).fillna(1)
            trades["pnl"] = (
                (trades["exit_price"] - trades["entry_price"]) * trades["volume"] * direction
            )
            analyzer = TradeAnalyzer(trades)
            pnl_by_dur = analyzer.pnl_by_duration()
            HOLD_REPORT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"duration_min": list(pnl_by_dur.keys()), "pnl": list(pnl_by_dur.values())}
            ).to_csv(HOLD_REPORT_DIR / "pnl_by_duration.csv", index=False)
        except Exception:
            logger.exception("Failed generating hold duration report")


def main() -> None:  # pragma: no cover - CLI entry point
    reprocess_trades()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
