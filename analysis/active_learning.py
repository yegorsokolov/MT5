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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_queue(self) -> list[dict]:
        if not self.queue_path.exists():
            return []
        try:
            data = json.loads(self.queue_path.read_text())
        except Exception:  # pragma: no cover - corrupted file
            logger.warning("Failed to read active learning queue, resetting it")
            return []
        if not isinstance(data, list):  # pragma: no cover - defensive guard
            return []
        return data

    def _write_queue(self, entries: list[dict]) -> None:
        self.queue_path.write_text(json.dumps(entries))

    def _load_labeled(self) -> pd.DataFrame:
        if not self.labeled_path.exists():
            return pd.DataFrame(columns=["id", "label"])
        try:
            return pd.read_json(self.labeled_path)
        except ValueError:  # pragma: no cover - corrupted file
            logger.warning("Failed to read labeled data file; discarding contents")
            return pd.DataFrame(columns=["id", "label"])

    def __len__(self) -> int:
        return len(self._load_queue())

    def size(self) -> int:
        """Return number of samples currently queued."""
        return len(self)

    # ------------------------------------------------------------------
    # Queue operations
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, int]:
        """Return a dictionary describing queue and label availability."""

        queued = len(self._load_queue())
        labeled = len(self._load_labeled())
        stats = {
            "queued": int(queued),
            "awaiting_label": int(queued),
            "ready_for_merge": int(labeled),
        }
        return stats

    def push(self, ids: Sequence[int], probs: np.ndarray, k: int = 10) -> None:
        """Push ``k`` most uncertain samples to the queue."""
        unc = entropy_uncertainty(probs)
        top = np.argsort(-unc)[:k]
        entries = [
            {"id": int(ids[i]), "uncertainty": float(unc[i]), "timestamp": time.time()}
            for i in top
        ]
        existing = self._load_queue()
        existing_ids = {int(item.get("id")) for item in existing}
        new_entries = [e for e in entries if e["id"] not in existing_ids]
        if not new_entries:
            logger.info("No new high-uncertainty samples to queue")
            return
        existing.extend(new_entries)
        self._write_queue(existing)
        logger.info("Queued %s samples for labeling", len(new_entries))
        logger.info("Queue stats after push: %s", self.stats())

    def push_low_confidence(
        self, ids: Sequence[int], probs: np.ndarray, threshold: float = 0.6
    ) -> int:
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
            return 0
        entries = [
            {
                "id": int(ids[i]),
                "confidence": float(max_conf[i]),
                "timestamp": time.time(),
            }
            for i in np.where(mask)[0]
        ]
        existing = self._load_queue()
        existing_ids = {int(item.get("id")) for item in existing}
        new_entries = [e for e in entries if e["id"] not in existing_ids]
        if not new_entries:
            logger.info("No new low-confidence samples met the threshold")
            return 0
        existing.extend(new_entries)
        self._write_queue(existing)
        logger.info("Queued %s low-confidence samples", len(new_entries))
        logger.info("Queue stats after push: %s", self.stats())
        return len(new_entries)

    def pop_labeled(self) -> pd.DataFrame:
        """Return any newly labeled samples and log turnaround time."""
        if not self.labeled_path.exists():
            return pd.DataFrame()
        labeled = self._load_labeled()
        try:
            self.labeled_path.unlink()
        except FileNotFoundError:  # pragma: no cover - race condition guard
            pass
        queue = pd.DataFrame(self._load_queue())
        if queue.empty:
            queue = pd.DataFrame(columns=["id", "timestamp"])
        merged = labeled.merge(queue, on="id", how="left")
        if not merged.empty:
            turnaround = time.time() - merged["timestamp"]
            logger.info(
                "Received %s labels (mean turnaround %.2fs)",
                len(merged),
                float(turnaround.mean()),
            )
        if not queue.empty:
            remaining = queue[~queue["id"].isin(labeled["id"])].copy()
            self._write_queue(remaining.to_dict(orient="records"))
            logger.info("Queue stats after label ingestion: %s", self.stats())
        return merged[["id", "label"]]


def merge_labels(
    df: pd.DataFrame, labeled: pd.DataFrame, label_col: str
) -> pd.DataFrame:
    """Merge ``labeled`` samples into ``df`` under ``label_col``."""
    if labeled.empty:
        return df

    df = df.copy()
    labeled_unique = labeled.drop_duplicates("id")
    updated = 0

    if "id" in df.columns:
        id_to_index = pd.Series(df.index, index=df["id"])
        target_indices = labeled_unique["id"].map(id_to_index)
        mask = target_indices.notna()
        if mask.any():
            df.loc[target_indices.loc[mask], label_col] = labeled_unique.loc[
                mask, "label"
            ].values
            updated = int(mask.sum())
    else:
        idx = labeled_unique["id"]
        try:
            idx = idx.astype(df.index.dtype, copy=False)
        except Exception:  # pragma: no cover - dtype coercion best effort
            pass
        mask = idx.isin(df.index)
        if mask.any():
            df.loc[idx[mask], label_col] = labeled_unique.loc[mask, "label"].values
            updated = int(mask.sum())

    if updated:
        logger.info("Merged %s verified labels into training set", updated)
    else:
        logger.warning(
            "Received %s labels but none matched training indices", len(labeled_unique)
        )
    return df
