from __future__ import annotations

"""Genetic programming based feature evolution."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from feature_store import register_feature as store_register_feature


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
    def _expression_to_code(self, expression: str, cols: List[str]) -> str:
        """Map program string to pandas-friendly expression."""

        code = expression
        for i, col in enumerate(cols):
            code = code.replace(f"X{i}", f"df['{col}']")
        return code

    def _append_to_module(self, feats: List[EvolvedFeature], module_path: Path) -> None:
        """Append evolved feature functions to ``module_path``."""

        module_path.parent.mkdir(parents=True, exist_ok=True)
        if not module_path.exists():
            module_path.write_text("# Auto-generated evolved features\n")
        with module_path.open("a") as fh:
            for feat in feats:
                code = self._expression_to_code(feat.expression, feat.base_columns)
                fh.write(f"\n# Evolved feature: {feat.name}\n")
                fh.write(f"def {feat.name}(df: pd.DataFrame) -> pd.Series:\n")
                fh.write(f"    return {code}\n")

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
        module_path: str | Path | None = None,
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

        # score candidates by cross-validated lift
        cv = KFold(n_splits=3, shuffle=True, random_state=0)
        model = LinearRegression()
        base_score = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
        lifts: List[float] = []
        for i in range(new_features.shape[1]):
            cand = np.column_stack([X, new_features[:, i]])
            score = cross_val_score(model, cand, y, cv=cv, scoring="r2").mean()
            lifts.append(score - base_score)

        # pair lifts with their corresponding programs and values
        candidates = [
            (lift, prog, new_features[:, idx])
            for idx, (lift, prog) in enumerate(zip(lifts, programs))
        ]
        candidates.sort(key=lambda t: t[0], reverse=True)

        manifest = self._load_manifest()
        start_idx = len(manifest)
        selected_feats: List[EvolvedFeature] = []
        for i, (lift, prog, values) in enumerate(candidates):
            name = f"gp_feat_{start_idx + i}"
            df[name] = values
            meta = EvolvedFeature(
                name=name,
                expression=str(prog),
                score=float(lift),
                base_columns=base_cols,
            )
            with open(self.store_dir / f"{name}.pkl", "wb") as fh:
                pickle.dump(prog, fh)
            manifest.append(meta.__dict__)
            selected_feats.append(meta)
            try:
                store_register_feature(
                    name,
                    pd.DataFrame({name: values}),
                    {"expression": meta.expression},
                )
            except Exception:
                pass

        if selected_feats and module_path is not None:
            self._append_to_module(selected_feats, Path(module_path))

        self._save_manifest(manifest)
        self._write_last_regime(current_regime)
        return df


__all__ = ["FeatureEvolver", "EvolvedFeature"]
