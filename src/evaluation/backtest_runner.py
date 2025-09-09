from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from backtest import run_backtest

METADATA_FILE = Path(__file__).resolve().parents[2] / "strategies" / "metadata.json"


def _load_metadata() -> Dict[str, Any]:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text())
    return {}


def _save_metadata(data: Dict[str, Any]) -> None:
    METADATA_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))


def update_metadata(strategy: str, metrics: Dict[str, Any]) -> None:
    """Update metadata entry for ``strategy`` with ``metrics``.

    Approval is granted if Sharpe ratio exceeds ``1.0``.
    """

    data = _load_metadata()
    record = data.get(strategy, {"status": "experimental", "metrics": {}, "approved": False})
    record["metrics"] = metrics
    sharpe = metrics.get("sharpe_ratio") or metrics.get("sharpe") or 0
    record["approved"] = sharpe > 1.0
    record["status"] = "approved" if record["approved"] else "experimental"
    data[strategy] = record
    _save_metadata(data)


def run_and_update(strategy: str, cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run a backtest for ``strategy`` and update its metadata."""

    cfg = cfg or {}
    metrics = run_backtest(cfg, external_strategy=f"strategies/{strategy}.py")
    update_metadata(strategy, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest and update metadata")
    parser.add_argument("strategy", help="Strategy name without extension")
    parser.add_argument("--config", type=str, help="Optional YAML config file")
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        import yaml

        cfg = yaml.safe_load(Path(args.config).read_text())

    metrics = run_and_update(args.strategy, cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
