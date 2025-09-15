"""Utilities for generating and evaluating pseudo labels."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_pseudo_labels(
    model,
    X: pd.DataFrame,
    y_true: Iterable[int] | None = None,
    threshold: float = 0.9,
    output_dir: Path | str = Path("data/pseudo_labels"),
    report_dir: Path | str = Path("reports/pseudo_label"),
) -> Path:
    """Generate pseudo labels for high-confidence predictions.

    Parameters
    ----------
    model:
        Trained model exposing ``predict_proba``.
    X:
        Feature matrix for which to generate pseudo labels.
    y_true:
        Optional true labels to compute precision/recall for analysis.
    threshold:
        Minimum predicted probability required to keep a pseudo label.
    output_dir:
        Directory where pseudo label files will be written.
    report_dir:
        Directory where evaluation metrics will be stored.

    Returns
    -------
    Path
        The path of the written pseudo label parquet file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    probs = model.predict_proba(X)
    preds = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)
    mask = max_probs >= threshold

    pseudo = X[mask].copy()
    pseudo["pseudo_label"] = preds[mask]

    out_path = output_dir / "pseudo_labels.csv"
    pseudo.to_csv(out_path, index=False)

    if y_true is not None:
        y_true = np.asarray(list(y_true))[mask]
        y_pred = preds[mask]
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        metrics = {"precision": precision, "recall": recall}
        (report_dir / "metrics.json").write_text(json.dumps(metrics))
        logger.info("Pseudo label precision=%s recall=%s", precision, recall)
    return out_path


def cli() -> None:
    """Command line interface for pseudo label generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to joblib model")
    parser.add_argument("--data", required=True, help="Path to feature parquet/csv")
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    import joblib

    model = joblib.load(args.model)
    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        X = pd.read_parquet(data_path)
    else:
        X = pd.read_csv(data_path)
    generate_pseudo_labels(model, X, threshold=args.threshold)


if __name__ == "__main__":
    cli()
