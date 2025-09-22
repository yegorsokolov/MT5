"""Plot feature importance computed by :mod:`mt5.train`."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mt5.log_utils import LOG_DIR
import argparse


def main(csv_path: str | None = None, top_n: int = 20) -> None:
    """Generate a bar chart of the top SHAP features."""
    path = Path(csv_path) if csv_path else LOG_DIR / "feature_importance.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run 'python -m mt5.train' first")
    df = pd.read_csv(path)
    df.sort_values("importance", ascending=False, inplace=True)
    top = df.head(top_n)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("mean(|SHAP value|)")
    plt.title("Top SHAP Features")
    plt.tight_layout()
    out = LOG_DIR / "shap_importance.png"
    plt.savefig(out)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SHAP feature importance")
    parser.add_argument("--csv", help="path to feature_importance.csv", default=None)
    parser.add_argument("--top", type=int, default=20, help="number of features to plot")
    args = parser.parse_args()
    main(args.csv, args.top)
