import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evolved_symbols", ROOT / "features" / "evolved_symbols.py"
)
evolved_symbols = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(evolved_symbols)


def _synthetic() -> pd.DataFrame:
    n = 50
    price = np.linspace(1.0, 2.0, n)
    volume = np.linspace(2.0, 3.0, n)
    return pd.DataFrame({"price": price, "volume": volume})


def test_deterministic_symbolic_features():
    df = _synthetic()
    out1 = evolved_symbols.compute(df)
    out2 = evolved_symbols.compute(df)
    assert out1["evo_sym_0"].equals(out2["evo_sym_0"])
