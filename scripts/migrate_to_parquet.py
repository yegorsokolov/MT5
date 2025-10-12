from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Ensure repository root is importable when executed via ``python scripts/...``
    sys.path.insert(0, str(REPO_ROOT))

from data.history import save_history_parquet


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    for csv in data_dir.glob("*.csv"):
        df = pd.read_csv(csv)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        out = csv.with_suffix(".parquet")
        save_history_parquet(df, out)
        print(f"Converted {csv.name} -> {out.name}")


if __name__ == "__main__":
    main()
