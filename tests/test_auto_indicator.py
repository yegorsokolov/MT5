import pandas as pd
from pathlib import Path
import importlib.util


def test_parameterised_generation_and_skip():
    spec = importlib.util.spec_from_file_location(
        "auto_indicator", Path(__file__).resolve().parents[1] / "features" / "auto_indicator.py"
    )
    auto_indicator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auto_indicator)  # type: ignore

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "target": [0, 1, 0, 1, 0],
        }
    )
    out = auto_indicator.compute(df, lags=[1, 2], windows=[3], skip=["target"])

    # lag and rolling features created for non-skipped columns
    assert {"a_lag1", "a_lag2", "b_lag1", "b_lag2"}.issubset(out.columns)
    assert {"a_mean3", "a_std3", "b_mean3", "b_std3"}.issubset(out.columns)

    # target column should not be transformed
    assert not any(col.startswith("target_lag") or col.startswith("target_mean") for col in out.columns)
    assert out.shape[0] == df.shape[0]
