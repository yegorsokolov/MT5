from __future__ import annotations

"""Command-line utility to evolve indicators and store formulas."""

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.indicator_evolution import IndicatorEvolver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV file containing training data")
    parser.add_argument("--target", default="target", help="Target column")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to store formulas. By default a versioned file "
            "evolved_indicators_v*.json is created under feature_store/"
        ),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    evolver = IndicatorEvolver()
    indicators = evolver.evolve(X, y)
    data = [ind.__dict__ for ind in indicators]

    if args.output is None:
        feature_dir = Path(__file__).resolve().parent / "feature_store"
        feature_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(feature_dir.glob("evolved_indicators_v*.json"))
        version = len(existing) + 1
        output = feature_dir / f"evolved_indicators_v{version}.json"
    else:
        output = args.output

    output.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
