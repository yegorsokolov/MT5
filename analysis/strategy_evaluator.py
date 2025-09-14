"""Evaluate registered algorithms on recent history by instrument/basket."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from strategy.router import StrategyRouter, FeatureDict
from analytics.metrics_aggregator import record_metric
from .performance_correlation import compute_correlations


@dataclass
class StrategyEvaluator:
    """Run rolling backtests for algorithms and store risk metrics.

    Parameters
    ----------
    window: int
        Number of most recent observations per market basket to include in the
        evaluation.  Acts as a rolling window length.
    history_path: Path | str
        Location of the historical features/returns dataset.  The file is
        expected to contain ``return``, ``volatility``, ``trend_strength``,
        ``instrument`` and ``market_basket`` columns.  By default
        ``data/history.parquet`` is used.
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
    def _risk_metrics(
        returns: Iterable[float], daily_limit: float, total_limit: float
    ) -> dict:
        """Compute basic risk metrics and drawdown limit flags."""
        arr = np.asarray(list(returns), dtype=float)
        if arr.size == 0:
            return {
                "pnl": 0.0,
                "sharpe": 0.0,
                "drawdown": 0.0,
                "daily_loss": 0.0,
                "daily_limit_violation": False,
                "total_limit_violation": False,
            }
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        sharpe = mean / (std + 1e-9)
        cumulative = (1 + arr).cumprod()
        drawdown = float((np.maximum.accumulate(cumulative) - cumulative).max())
        worst_daily = float(np.minimum.reduce(np.clip(arr, a_max=0, a_min=None)))
        daily_loss = -worst_daily
        daily_violation = daily_loss > daily_limit
        total_violation = drawdown > total_limit
        return {
            "pnl": float(arr.sum()),
            "sharpe": sharpe,
            "drawdown": drawdown,
            "daily_loss": daily_loss,
            "daily_limit_violation": daily_violation,
            "total_limit_violation": total_violation,
        }

    # ------------------------------------------------------------------
    def evaluate(
        self,
        history: pd.DataFrame,
        router: StrategyRouter,
        daily_limit: float = float("inf"),
        total_limit: float = float("inf"),
    ) -> pd.DataFrame:
        """Evaluate ``router`` algorithms on ``history`` and persist scoreboard."""
        if history.empty:
            return pd.DataFrame(
                columns=["pnl", "sharpe", "drawdown"],
                index=pd.MultiIndex.from_tuples(
                    [], names=["instrument", "market_basket", "algorithm"]
                ),
            )

        records: List[dict] = []
        corr_records: List[dict] = []
        for (inst, basket), df_basket in history.groupby(["instrument", "market_basket"]):
            df_window = df_basket.tail(self.window)
            feats = df_window[["volatility", "trend_strength", "regime"]].to_dict(
                "records"
            )
            rets = df_window["return"].values
            corr_features = [
                c
                for c in df_window.columns
                if c not in ["return", "market_basket", "instrument", "regime"]
            ]
            for name, algo in router.algorithms.items():
                actions = [
                    algo({**f, "market_basket": basket, "instrument": inst}) for f in feats
                ]
                pnl = np.asarray(actions) * rets
                metrics = self._risk_metrics(pnl, daily_limit, total_limit)
                records.append(
                    {"instrument": inst, "market_basket": basket, "algorithm": name, **metrics}
                )
                # Correlations
                corr_df = compute_correlations(df_window[corr_features], pnl, corr_features)
                ts = pd.Timestamp.utcnow()
                for row in corr_df.itertuples(index=False):
                    corr_records.append(
                        {
                            "timestamp": ts,
                            "instrument": inst,
                            "market_basket": basket,
                            "algorithm": name,
                            "feature": row.feature,
                            "pearson": row.pearson,
                            "spearman": row.spearman,
                        }
                    )

        scoreboard = pd.DataFrame(records).set_index(
            ["instrument", "market_basket", "algorithm"]
        )
        router.scoreboard = scoreboard
        try:
            path = Path("reports/strategy_scores.parquet")
            router.scoreboard_path = path
            path.parent.mkdir(parents=True, exist_ok=True)
            router.scoreboard.to_parquet(path)
        except Exception:
            # Optional parquet dependencies may be missing in minimal setups.
            pass

        # Log metrics for dashboard consumption
        for _, row in scoreboard.reset_index().iterrows():
            inst = row["instrument"]
            basket = row["market_basket"]
            alg = row["algorithm"]
            try:
                record_metric(
                    "strategy_eval_pnl",
                    float(row["pnl"]),
                    tags={"algorithm": alg, "instrument": inst, "basket": basket},
                )
                record_metric(
                    "strategy_eval_drawdown",
                    float(row["drawdown"]),
                    tags={"algorithm": alg, "instrument": inst, "basket": basket},
                )
            except Exception:
                pass

        # Append correlation results
        if corr_records:
            corr_path = Path("reports/performance_correlations.parquet")
            try:
                existing = pd.read_parquet(corr_path) if corr_path.exists() else pd.DataFrame()
            except Exception:
                existing = pd.DataFrame()
            new_corr = pd.DataFrame(corr_records)
            try:
                corr_path.parent.mkdir(parents=True, exist_ok=True)
                pd.concat([existing, new_corr], ignore_index=True).to_parquet(corr_path)
            except Exception:
                pass

        return scoreboard

    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Load history and evaluate using a fresh :class:`StrategyRouter`."""
        history = self.load_history()
        router = StrategyRouter()
        return self.evaluate(history, router)


__all__ = ["StrategyEvaluator"]
