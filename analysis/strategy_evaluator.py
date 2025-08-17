"""Evaluate registered algorithms on recent history by regime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from strategy.router import StrategyRouter, FeatureDict


@dataclass
class StrategyEvaluator:
    """Run rolling backtests for algorithms and store risk metrics.

    Parameters
    ----------
    window: int
        Number of most recent observations per regime to include in the
        evaluation.  Acts as a rolling window length.
    history_path: Path | str
        Location of the historical features/returns dataset.  The file is
        expected to contain ``return``, ``volatility``, ``trend_strength`` and
        ``regime`` columns.  By default ``data/history.parquet`` is used.
    """

    window: int = 252
    history_path: Path | str = Path("data/history.parquet")

    # ------------------------------------------------------------------
    def load_history(self) -> pd.DataFrame:
        """Return the historical dataset or an empty dataframe if missing."""
        path = Path(self.history_path)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    @staticmethod
    def _risk_metrics(returns: Iterable[float]) -> dict:
        arr = np.asarray(list(returns), dtype=float)
        if arr.size == 0:
            return {"sharpe": 0.0, "drawdown": 0.0}
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        sharpe = mean / (std + 1e-9)
        cumulative = (1 + arr).cumprod()
        drawdown = float((np.maximum.accumulate(cumulative) - cumulative).max())
        return {"sharpe": sharpe, "drawdown": drawdown}

    # ------------------------------------------------------------------
    def evaluate(self, history: pd.DataFrame, router: StrategyRouter) -> pd.DataFrame:
        """Evaluate ``router`` algorithms on ``history`` and persist scoreboard."""
        if history.empty:
            return pd.DataFrame(
                columns=["sharpe", "drawdown"],
                index=pd.MultiIndex.from_tuples([], names=["regime", "algorithm"]),
            )

        records = []
        for regime, df_reg in history.groupby("regime"):
            df_window = df_reg.tail(self.window)
            feats = df_window[["volatility", "trend_strength"]].to_dict("records")
            rets = df_window["return"].values
            for name, algo in router.algorithms.items():
                actions = [algo({**f, "regime": regime}) for f in feats]
                pnl = np.asarray(actions) * rets
                metrics = self._risk_metrics(pnl)
                records.append({"regime": regime, "algorithm": name, **metrics})

        scoreboard = pd.DataFrame(records).set_index(["regime", "algorithm"])
        router.scoreboard = scoreboard
        try:
            router.scoreboard_path.parent.mkdir(parents=True, exist_ok=True)
            router.scoreboard.to_parquet(router.scoreboard_path)
        except Exception:
            # Optional parquet dependencies may be missing in minimal setups.
            pass
        return scoreboard

    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Load history and evaluate using a fresh :class:`StrategyRouter`."""
        history = self.load_history()
        router = StrategyRouter()
        return self.evaluate(history, router)


__all__ = ["StrategyEvaluator"]
