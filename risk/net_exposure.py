from __future__ import annotations

"""Track aggregate long and short notional exposure."""

from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import threading
from typing import Dict

import numpy as np
import pandas as pd
from analytics.metrics_store import record_metric


@dataclass
class NetExposure:
    """Maintain long/short exposures and enforce portfolio limits."""

    max_long: float = float("inf")
    max_short: float = float("inf")
    long: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    short: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    window: int = 20
    corr: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:  # pragma: no cover - simple init
        self._returns: Dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self._lock = threading.Lock()
        self._log_dir = Path("logs")
        self._load_state()

    # ------------------------------------------------------------------
    def _totals(self) -> tuple[float, float]:
        long_vec = {s: v for s, v in self.long.items() if v > 0}
        short_vec = {s: v for s, v in self.short.items() if v > 0}
        return self._weighted_exposure(long_vec), self._weighted_exposure(short_vec)

    # ------------------------------------------------------------------
    def _weighted_exposure(self, exposures: Dict[str, float]) -> float:
        if not exposures:
            return 0.0
        symbols = list(exposures)
        vec = np.array([exposures[s] for s in symbols])
        if self.corr.empty:
            corr = np.eye(len(symbols))
        else:
            corr = (
                self.corr.reindex(index=symbols, columns=symbols)
                .fillna(0)
                .to_numpy()
            )
            np.fill_diagonal(corr, 1.0)
        return float(np.sqrt(vec @ corr @ vec))

    # ------------------------------------------------------------------
    def limit(self, symbol: str, notional: float) -> float:
        """Return notional allowed for a proposed trade."""

        with self._lock:
            if notional > 0:
                return self._limit_trade(symbol, notional, self.long, self.max_long)
            elif notional < 0:
                allowed = self._limit_trade(
                    symbol, -notional, self.short, self.max_short
                )
                return -allowed
            return 0.0

    # ------------------------------------------------------------------
    def _limit_trade(
        self, symbol: str, notional: float, book: Dict[str, float], cap: float
    ) -> float:
        symbols = list(book)
        if symbol not in symbols:
            symbols.append(symbol)
        vec = np.array([book.get(s, 0.0) for s in symbols])
        idx = symbols.index(symbol)
        if self.corr.empty:
            corr = np.eye(len(symbols))
        else:
            corr = (
                self.corr.reindex(index=symbols, columns=symbols)
                .fillna(0)
                .to_numpy()
            )
            np.fill_diagonal(corr, 1.0)
        current_sq = float(vec @ corr @ vec)
        if current_sq >= cap**2:
            return 0.0
        row = corr[idx]
        a = row[idx]
        b = 2 * np.dot(row, vec)
        c = current_sq - cap**2
        disc = b**2 - 4 * a * c
        if disc <= 0:
            return 0.0
        x_max = (-b + float(np.sqrt(disc))) / (2 * a)
        return min(notional, x_max)

    # ------------------------------------------------------------------
    def update(self, symbol: str, notional: float) -> None:
        """Record executed trade notional for ``symbol``."""

        with self._lock:
            if notional > 0:
                self.long[symbol] += notional
            elif notional < 0:
                self.short[symbol] += -notional
            long_tot, short_tot = self._totals()
            try:
                record_metric("long_exposure", long_tot)
                record_metric("short_exposure", short_tot)
            except Exception:
                pass
            self._persist()

    # ------------------------------------------------------------------
    def totals(self) -> Dict[str, float]:
        """Return dictionary of long, short and net exposure."""

        with self._lock:
            long_tot, short_tot = self._totals()
        return {"long": long_tot, "short": short_tot, "net": long_tot - short_tot}

    # ------------------------------------------------------------------
    def _persist(self) -> None:
        """Persist exposure books to parquet for crash recovery."""

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                list(self.long.items()), columns=["symbol", "notional"]
            ).to_parquet(self._log_dir / "net_exposure_long.parquet")
            pd.DataFrame(
                list(self.short.items()), columns=["symbol", "notional"]
            ).to_parquet(self._log_dir / "net_exposure_short.parquet")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        """Load exposure books from parquet if available."""

        try:
            long_path = self._log_dir / "net_exposure_long.parquet"
            short_path = self._log_dir / "net_exposure_short.parquet"
            if long_path.exists():
                df_long = pd.read_parquet(long_path)
                self.long.update(
                    df_long.set_index("symbol")["notional"].to_dict()
                )
            if short_path.exists():
                df_short = pd.read_parquet(short_path)
                self.short.update(
                    df_short.set_index("symbol")["notional"].to_dict()
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    def record_returns(self, returns: Dict[str, float]) -> None:
        """Update rolling correlation matrix from symbol returns."""

        for sym, ret in returns.items():
            self._returns[sym].append(ret)
        if not self._returns:
            return
        df = pd.DataFrame(self._returns)
        self.corr = df.corr().fillna(0)
        try:
            vals = self.corr.values
            if vals.size > 1:
                avg_corr = float(
                    vals[np.triu_indices_from(vals, k=1)].mean()
                )
                record_metric("avg_correlation", avg_corr)
        except Exception:
            pass
