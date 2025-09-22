from __future__ import annotations

"""Command-line utility to evolve symbolic feature formulas and store as YAML."""

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.symbolic_features import SymbolicFeatureEvolver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV file containing training data")
    parser.add_argument("--target", default="target", help="Target column")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/evolved_symbols.yaml"),
        help="Path to store formulas",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    evolver = SymbolicFeatureEvolver()
    symbols = evolver.evolve(X, y)
    data = [s.__dict__ for s in symbols]
    args.output.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
