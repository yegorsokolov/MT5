"""Train a PriceDistributionModel and report distributional metrics."""

from __future__ import annotations

import argparse
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from models.price_distribution import PriceDistributionModel

logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract numeric feature matrix and return column from ``df``."""

    if "return" not in df.columns:
        raise ValueError("DataFrame must contain a 'return' column")
    y = df["return"].to_numpy()
    X = df.drop(columns=["return"]).select_dtypes(include=[np.number]).to_numpy()
    return X, y


def train_price_distribution(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_components: int = 3,
    epochs: int = 100,
) -> Tuple[PriceDistributionModel, Dict[str, float]]:
    """Fit ``PriceDistributionModel`` and evaluate coverage and shortfall."""

    model = PriceDistributionModel(input_dim=X_train.shape[1], n_components=n_components)
    model.fit(X_train, y_train, epochs=epochs)

    lower = model.percentile(X_val, 0.05)
    upper = model.percentile(X_val, 0.95)
    coverage = float(np.mean((y_val >= lower) & (y_val <= upper)))
    es = float(model.expected_shortfall(X_val, 0.05).mean())

    mu, sigma = float(y_train.mean()), float(y_train.std(ddof=0))
    baseline_lower = mu + sigma * -1.6448536269514729
    baseline_upper = mu + sigma * 1.6448536269514722
    baseline_cov = float(np.mean((y_val >= baseline_lower) & (y_val <= baseline_upper)))

    logger.info("Coverage: %.3f (baseline %.3f)", coverage, baseline_cov)
    logger.info("Expected shortfall: %.6f", es)

    metrics = {
        "coverage": coverage,
        "baseline_coverage": baseline_cov,
        "expected_shortfall": es,
    }
    return model, metrics


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="CSV file for training data")
    parser.add_argument("--val", required=True, help="CSV file for validation data")
    parser.add_argument("--n-components", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    train_price_distribution(
        X_train,
        y_train,
        X_val,
        y_val,
        n_components=args.n_components,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

