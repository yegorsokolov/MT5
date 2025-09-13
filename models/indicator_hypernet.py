from __future__ import annotations

"""Hypernetwork for on-the-fly indicator discovery.

This lightweight implementation deterministically generates simple
indicator formulas conditioned on summary statistics of the asset features
and current market regime.  Generated formulas are persisted to
``feature_store/indicator_formulas.json`` so that future runs can reload and
reuse the same indicators.  Basic performance metrics are logged in
``indicator_performance.jsonl`` allowing downstream analysis to track
indicator utility over time.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class IndicatorRecord:
    """Metadata for a generated indicator."""

    name: str
    formula: str


class IndicatorHyperNet:
    """Deterministic hypernetwork producing indicator formulas."""

    def __init__(self, store_dir: str | Path = "feature_store") -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.formula_path = self.store_dir / "indicator_formulas.json"
        self.log_path = self.store_dir / "indicator_performance.jsonl"
        self.formulas: Dict[str, str] = {}
        if self.formula_path.exists():
            try:
                self.formulas = json.loads(self.formula_path.read_text())
            except Exception:  # pragma: no cover - corrupted file
                self.formulas = {}

    # ------------------------------------------------------------------
    def _state_seed(self, df: pd.DataFrame) -> int:
        """Derive a deterministic seed from the feature state."""

        regime = int(df["market_regime"].iloc[-1]) if "market_regime" in df.columns else 0
        stats = df.select_dtypes(include=[np.number]).mean().sum()
        return int(abs(float(stats) * 1e6)) + regime

    def _generate(self, df: pd.DataFrame) -> IndicatorRecord:
        """Create a new rolling-mean indicator conditioned on ``df``."""

        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "market_regime"]
        base = num_cols[0]
        seed = self._state_seed(df)
        rng = np.random.RandomState(seed)
        window = int(rng.randint(2, 10))
        name = f"hn_ma_{window}"
        formula = f"df['{base}'].rolling(window={window}, min_periods=1).mean()"
        return IndicatorRecord(name=name, formula=formula)

    def _save_formulas(self) -> None:
        self.formula_path.write_text(json.dumps(self.formulas, indent=2))

    # ------------------------------------------------------------------
    def apply_or_generate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Apply stored indicators or generate a new one.

        Returns the augmented ``df`` and a mapping of newly generated
        indicators to their formulas.
        """

        generated: Dict[str, str] = {}
        for name, formula in self.formulas.items():
            try:
                df[name] = eval(formula, {"np": np, "pd": pd}, {"df": df})
            except Exception:  # pragma: no cover - best effort
                continue

        if not self.formulas:
            rec = self._generate(df)
            self.formulas[rec.name] = rec.formula
            self._save_formulas()
            df[rec.name] = eval(rec.formula, {"np": np, "pd": pd}, {"df": df})
            generated[rec.name] = rec.formula

        return df, generated

    # ------------------------------------------------------------------
    def log_performance(self, name: str, score: float) -> None:
        """Append ``score`` for indicator ``name`` to the log file."""

        entry = {"name": name, "score": float(score)}
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")


__all__ = ["IndicatorHyperNet"]

