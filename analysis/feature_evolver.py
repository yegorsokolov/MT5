from __future__ import annotations

"""Genetic programming based feature evolution."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer


@dataclass
class EvolvedFeature:
    """Represents a single evolved feature."""

    name: str
    expression: str
    score: float
    base_columns: List[str]


class FeatureEvolver:
    """Evolve mathematical features using genetic programming.

    The evolver stores generated expression trees and their performance
    statistics in ``store_dir`` so that later runs can reproduce features.
    Evolution is triggered only when the market regime changes to avoid
    excessive computation.
    """

    def __init__(self, store_dir: str | Path | None = None) -> None:
        if store_dir is None:
            store_dir = Path(__file__).resolve().parent.parent / "feature_store"
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.store_dir / "manifest.json"
        self.regime_file = self.store_dir / "last_regime.txt"

    # ------------------------------------------------------------------
    def _load_manifest(self) -> List[Dict]:
        if self.manifest_file.exists():
            try:
                return json.loads(self.manifest_file.read_text())
            except Exception:
                return []
        return []

    def _save_manifest(self, data: List[Dict]) -> None:
        self.manifest_file.write_text(json.dumps(data, indent=2))

    def _read_last_regime(self) -> int:
        if self.regime_file.exists():
            try:
                return int(self.regime_file.read_text())
            except Exception:
                return -1
        return -1

    def _write_last_regime(self, regime: int) -> None:
        self.regime_file.write_text(str(regime))

    # ------------------------------------------------------------------
    def apply_stored_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously evolved features from the manifest to ``df``."""

        manifest = self._load_manifest()
        for feat in manifest:
            prog_path = self.store_dir / f"{feat['name']}.pkl"
            if not prog_path.exists():
                continue
            try:
                with open(prog_path, "rb") as fh:
                    prog = pickle.load(fh)
                X = df[feat["base_columns"]].to_numpy()
                df[feat["name"]] = prog.execute(X)
            except Exception:
                continue
        return df

    # ------------------------------------------------------------------
    def maybe_evolve(
        self,
        df: pd.DataFrame,
        target_col: str,
        regime_col: str = "market_regime",
        generations: int = 2,
        population_size: int = 50,
        n_components: int = 3,
    ) -> pd.DataFrame:
        """Run evolution if the market regime has changed."""

        current_regime = int(df[regime_col].iloc[-1]) if regime_col in df.columns else 0
        if current_regime == self._read_last_regime():
            return df

        base_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in {target_col, regime_col}
        ]
        if len(base_cols) < 2 or target_col not in df.columns:
            self._write_last_regime(current_regime)
            return df

        X = df[base_cols].to_numpy()
        y = df[target_col].to_numpy()

        try:
            gp = SymbolicTransformer(
                generations=generations,
                population_size=population_size,
                hall_of_fame=n_components,
                n_components=n_components,
                random_state=0,
                n_jobs=1,
            )
            gp.fit(X, y)
        except Exception:
            self._write_last_regime(current_regime)
            return df

        new_features = gp.transform(X)
        programs = gp._best_programs[:n_components]
        manifest = self._load_manifest()
        start_idx = len(manifest)
        for i, prog in enumerate(programs):
            name = f"gp_feat_{start_idx + i}"
            df[name] = new_features[:, i]
            meta = EvolvedFeature(
                name=name,
                expression=str(prog),
                score=float(getattr(prog, "raw_fitness_", 0.0)),
                base_columns=base_cols,
            )
            with open(self.store_dir / f"{name}.pkl", "wb") as fh:
                pickle.dump(prog, fh)
            manifest.append(meta.__dict__)

        self._save_manifest(manifest)
        self._write_last_regime(current_regime)
        return df


__all__ = ["FeatureEvolver", "EvolvedFeature"]
