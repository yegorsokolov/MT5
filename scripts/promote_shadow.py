#!/usr/bin/env python3
"""Promote shadow strategies to active when performance improves."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _load_avg(path: Path, window: int) -> Optional[float]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df.tail(window)["pnl"].mean())


def promote(name: str, window: int, threshold: float) -> None:
    """Promote shadow strategy ``name`` if its average PnL improves enough."""
    shadow_path = Path("reports/shadow") / f"{name}.csv"
    active_path = Path("reports/active") / f"{name}.csv"
    shadow_avg = _load_avg(shadow_path, window)
    active_avg = _load_avg(active_path, window)
    if shadow_avg is None or active_avg is None:
        print("missing data for", name)
        return
    if shadow_avg > active_avg * (1 + threshold):
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(shadow_path.read_text())
        print(f"Promoted {name}: {shadow_avg:.4f} > {active_avg:.4f}")
    else:
        print(f"No promotion for {name}: {shadow_avg:.4f} <= {active_avg:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="strategy name")
    parser.add_argument("--window", type=int, default=100, help="lookback window")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="required fractional improvement over active",
    )
    args = parser.parse_args()
    promote(args.name, args.window, args.threshold)


if __name__ == "__main__":
    main()

