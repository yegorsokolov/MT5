from __future__ import annotations

"""Update risk factor models on a rolling window.

This module periodically refits either a principal component based factor
model or a macro-economic regression model on the most recent history of
strategy returns.  The resulting factor returns and exposures are stored with a
timestamp so downstream components can reconstruct the state at any point in
time.  After estimation the latest exposures are fed to the
:mod:`risk_manager` to refresh risk budgets and to the strategy router so
that contextual bandits can incorporate the new factors.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import logging
import pandas as pd
import numpy as np

from analysis.factor_model import FactorModel
from mt5.risk_manager import risk_manager

try:  # optional dependency – macro based model only
    from data.macro_features import load_macro_series
except Exception:  # pragma: no cover - used only when available
    def load_macro_series(symbols: Iterable[str]) -> pd.DataFrame:  # type: ignore
        return pd.DataFrame()

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/factors")


@dataclass
class FactorUpdater:
    """Refit factor models on a sliding window of returns."""

    window: int = 252
    model: str = "pca"
    n_factors: int = 3
    returns_path: Path | str = Path("data/returns.parquet")
    macro_symbols: List[str] = field(default_factory=list)

    def load_returns(self) -> pd.DataFrame:
        """Load historical strategy returns.

        The loader tries a couple of common file formats (Parquet then CSV) and
        gracefully falls back to an empty frame when no data is available.  This
        keeps the updater robust when running in minimal environments such as
        unit tests.
        """

        path = Path(self.returns_path)
        if not path.exists():
            # try a CSV next to the parquet
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                path = csv_path
            else:
                logger.warning("No returns file found at %s", path)
                return pd.DataFrame()
        try:
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
        except Exception:
            logger.exception("Failed loading returns from %s", path)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    def _fit_pca(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fm = FactorModel(n_factors=self.n_factors).fit(returns)
        return fm.get_factor_returns(), fm.get_exposures()

    def _fit_macro(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        macro_df = load_macro_series(self.macro_symbols)
        if macro_df.empty:
            logger.warning("No macro series available – falling back to PCA model")
            return self._fit_pca(returns)
        macro_df = macro_df.set_index("Date") if "Date" in macro_df.columns else macro_df
        df = returns.join(macro_df, how="inner").dropna()
        if df.empty:
            logger.warning("Unable to align returns with macro series – falling back to PCA")
            return self._fit_pca(returns)
        factors = df[macro_df.columns]
        X = factors.to_numpy(dtype=float)
        exposures = {}
        for col in returns.columns:
            y = df[col].to_numpy(dtype=float)
            if len(y) != len(X):
                continue
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            exposures[col] = beta
        exp_df = pd.DataFrame(exposures, index=factors.columns).T
        return factors, exp_df

    # ------------------------------------------------------------------
    def _store(self, factor_returns: pd.DataFrame, exposures: pd.DataFrame) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        try:
            factor_returns.to_csv(DATA_DIR / f"factor_returns_{ts}.csv")
            exposures.to_csv(DATA_DIR / f"exposures_{ts}.csv")
        except Exception:
            logger.exception("Failed writing factor matrices to %s", DATA_DIR)

    def _refresh_risk(self, exposures: pd.DataFrame) -> None:
        try:
            old = risk_manager.budget_allocator.budgets.copy()
            exp_dict: Dict[str, pd.Series] = {
                asset: exposures.loc[asset] for asset in exposures.index
            }
            new = risk_manager.rebalance_budgets(factor_exposures=exp_dict)
            for strat, budget in new.items():
                if abs(old.get(strat, 0.0) - budget) > 1e-12:
                    logger.info(
                        "Budget updated for %s: %.6f -> %.6f", strat, old.get(strat, 0.0), budget
                    )
        except Exception:
            logger.exception("Risk budget refresh failed")

    def _refresh_router(self, factor_names: Iterable[str]) -> None:
        try:
            from mt5.signal_queue import _ROUTER

            _ROUTER.set_factor_names(list(factor_names))
            logger.info("Strategy router factor names refreshed: %s", list(factor_names))
        except Exception:
            logger.exception("Strategy router refresh failed")

    # ------------------------------------------------------------------
    def run(self) -> None:
        returns = self.load_returns()
        if returns.empty:
            logger.warning("Skipping factor update – no returns available")
            return
        returns = returns.tail(self.window)
        if self.model.lower() == "macro":
            factors, exposures = self._fit_macro(returns)
        else:
            factors, exposures = self._fit_pca(returns)
        self._store(factors, exposures)
        self._refresh_router(factors.columns)
        self._refresh_risk(exposures)


def update_factors() -> None:
    """Convenience wrapper used by the scheduler."""
    FactorUpdater().run()


__all__ = ["FactorUpdater", "update_factors"]
