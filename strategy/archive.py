"""Persistent archive for generated trading strategies.

The archive stores every candidate strategy and its metadata as JSON records in
``reports/strategies`` so operators – and automated training routines – can
inspect, replay or fine-tune historic ideas.  Each entry is assigned a stable
identifier and a short JSON Lines catalog is maintained for lightweight
queries.

Strategies exceeding the minimum profit per unit of daily drawdown risk are
flagged as *important* and mirrored under ``reports/strategies/important`` for
easy discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping
import uuid

try:  # Prefer the shared loader so user-configured risk limits are honoured.
    from mt5.state_manager import load_user_risk as _load_user_risk
except Exception:  # pragma: no cover - fallback for stripped-down test envs.
    def _load_user_risk() -> Mapping[str, Any]:  # type: ignore
        return {"daily_drawdown": 0.049}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, Iterable):
        return [_normalise(v) for v in value]
    return repr(value)


def _to_percent(value: float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    if -1.0 <= numeric <= 1.0:
        return numeric * 100.0
    return numeric


@dataclass
class StrategyArchive:
    """Persist strategy payloads and surface high-performing candidates."""

    root: Path = Path("reports") / "strategies"
    important_subdir: str = "important"
    catalog_name: str = "catalog.jsonl"
    risk_loader: Callable[[], Mapping[str, Any]] = _load_user_risk

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.important_dir = self.root / self.important_subdir
        self.important_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / self.catalog_name

    # ------------------------------------------------------------------
    def record(
        self,
        strategy: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Persist ``strategy`` and return the path to the JSON snapshot."""

        entry_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        payload = {
            "id": entry_id,
            "timestamp": _now(),
            "strategy": _normalise(strategy),
            "metadata": _normalise(metadata or {}),
        }
        payload["important"] = bool(self._is_important(strategy, metadata))

        entry_path = self.root / f"{entry_id}.json"
        entry_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

        with self.catalog_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")

        if payload["important"]:
            important_path = self.important_dir / entry_path.name
            important_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

        return entry_path

    # ------------------------------------------------------------------
    def iter_entries(self) -> Iterator[dict[str, Any]]:
        """Yield archived strategies in insertion order."""

        if not self.catalog_path.exists():
            return iter(())  # type: ignore[return-value]

        def _iterator() -> Iterator[dict[str, Any]]:
            with self.catalog_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

        return _iterator()

    # ------------------------------------------------------------------
    def _is_important(
        self,
        strategy: Mapping[str, Any],
        metadata: Mapping[str, Any] | None,
    ) -> bool:
        try:
            risk = float(self.risk_loader().get("daily_drawdown", 0.0))
        except Exception:  # pragma: no cover - defensive fallback
            risk = 0.0
        risk_percent = _to_percent(risk)
        if risk_percent <= 0:
            return False

        profit_candidates: list[float] = []
        if metadata:
            for key in ("monthly_profit", "profit", "pnl"):
                value = metadata.get(key)
                if value is not None:
                    profit_candidates.append(_to_percent(value))
        if not profit_candidates:
            raw = strategy.get("monthly_profit") or strategy.get("pnl")
            profit_candidates.append(_to_percent(raw))

        profit_percent = max((v for v in profit_candidates if v > 0), default=0.0)
        if profit_percent <= 0:
            return False
        return profit_percent / risk_percent > 1.0


__all__ = ["StrategyArchive"]

