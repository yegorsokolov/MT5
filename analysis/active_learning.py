"""Active learning utilities for identifying and incorporating uncertain samples.

This module implements a lightweight active learning loop. During training,
models can queue up high-uncertainty samples for human or automatic
labelling. Returned labels are merged back into the dataset on the next
training run.

The queue is backed by JSON files on disk so it works in simple local
setups without additional services.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
QUEUE_PATH = DATA_DIR / "label_queue.json"
LABELED_PATH = DATA_DIR / "labeled_data.json"


def entropy_uncertainty(probs: np.ndarray) -> np.ndarray:
    """Return entropy of class probabilities."""
    if probs.ndim == 1:
        probs = np.vstack([1 - probs, probs]).T
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    return -np.sum(probs * np.log(probs), axis=1)


class ActiveLearningQueue:
    """Simple file-backed queue for active learning samples."""

    def __init__(
        self, queue_path: Path = QUEUE_PATH, labeled_path: Path = LABELED_PATH
    ) -> None:
        self.queue_path = queue_path
        self.labeled_path = labeled_path
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        if self.queue_path.exists():
            try:
                return len(json.loads(self.queue_path.read_text()))
            except Exception:  # pragma: no cover - corrupted file
                return 0
        return 0

    def size(self) -> int:
        """Return number of samples currently queued."""
        return len(self)

    def push(self, ids: Sequence[int], probs: np.ndarray, k: int = 10) -> None:
        """Push ``k`` most uncertain samples to the queue."""
        unc = entropy_uncertainty(probs)
        top = np.argsort(-unc)[:k]
        entries = [
            {"id": int(ids[i]), "uncertainty": float(unc[i]), "timestamp": time.time()}
            for i in top
        ]
        existing: list[dict] = []
        if self.queue_path.exists():
            try:
                existing = json.loads(self.queue_path.read_text())
            except Exception:  # pragma: no cover - corrupted file
                existing = []
        existing.extend(entries)
        self.queue_path.write_text(json.dumps(existing))
        logger.info("Queued %s samples for labeling", len(entries))

    def push_low_confidence(
        self, ids: Sequence[int], probs: np.ndarray, threshold: float = 0.6
    ) -> None:
        """Queue samples whose max predicted probability is below ``threshold``.

        Parameters
        ----------
        ids:
            Identifiers corresponding to ``probs`` rows.
        probs:
            Array of class probabilities for each sample.
        threshold:
            Maximum probability required to *avoid* queuing. Samples with
            confidence below this value are pushed to the queue.
        """

        if probs.ndim == 1:
            probs = np.vstack([1 - probs, probs]).T
        max_conf = probs.max(axis=1)
        mask = max_conf < threshold
        if not np.any(mask):
            return
        entries = [
            {
                "id": int(ids[i]),
                "confidence": float(max_conf[i]),
                "timestamp": time.time(),
            }
            for i in np.where(mask)[0]
        ]
        existing: list[dict] = []
        if self.queue_path.exists():
            try:
                existing = json.loads(self.queue_path.read_text())
            except Exception:  # pragma: no cover - corrupted file
                existing = []
        existing.extend(entries)
        self.queue_path.write_text(json.dumps(existing))
        logger.info("Queued %s low-confidence samples", len(entries))

    def pop_labeled(self) -> pd.DataFrame:
        """Return any newly labeled samples and log turnaround time."""
        if not self.labeled_path.exists():
            return pd.DataFrame()
        labeled = pd.read_json(self.labeled_path)
        self.labeled_path.unlink()
        queue = (
            pd.read_json(self.queue_path, convert_dates=False)
            if self.queue_path.exists()
            else pd.DataFrame(columns=["id", "timestamp"])
        )
        merged = labeled.merge(queue, on="id", how="left")
        if not merged.empty:
            turnaround = time.time() - merged["timestamp"]
            logger.info(
                "Received %s labels (mean turnaround %.2fs)",
                len(merged),
                float(turnaround.mean()),
            )
        if not queue.empty:
            remaining = queue[~queue["id"].isin(labeled["id"])]
            self.queue_path.write_text(remaining.to_json(orient="records"))
        return merged[["id", "label"]]


def merge_labels(
    df: pd.DataFrame, labeled: pd.DataFrame, label_col: str
) -> pd.DataFrame:
    """Merge ``labeled`` samples into ``df`` under ``label_col``."""
    if labeled.empty:
        return df
    df.loc[labeled["id"], label_col] = labeled["label"].values
    return df
