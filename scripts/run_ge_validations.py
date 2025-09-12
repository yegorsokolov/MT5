import pandas as pd
import runpy
from pathlib import Path

validate_ge = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "features" / "validators.py")
)["validate_ge"]


def main() -> None:
    validate_ge(pd.DataFrame({"price": [1, 2, 3]}), "prices")
    validate_ge(pd.DataFrame({"feature": [0.1, 0.2, 0.3]}), "features")
    validate_ge(pd.DataFrame({"Label": [-1, 0, 1]}), "labels")


if __name__ == "__main__":
    main()
