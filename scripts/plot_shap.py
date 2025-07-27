"""Plot feature importance from SHAP values."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from log_utils import LOG_DIR


def main(csv_path: str | None = None) -> None:
    path = Path(csv_path) if csv_path else LOG_DIR / "feature_importance.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run train.py first")
    df = pd.read_csv(path)
    df.sort_values("importance", ascending=False, inplace=True)
    top = df.head(20)
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
    main()
