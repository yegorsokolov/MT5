from __future__ import annotations

"""Command-line utility to evolve indicators and store formulas as YAML."""

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
        default=Path("analysis/evolved_indicators.yaml"),
        help="Path to store formulas",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    evolver = IndicatorEvolver()
    indicators = evolver.evolve(X, y)
    data = [ind.__dict__ for ind in indicators]
    args.output.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
