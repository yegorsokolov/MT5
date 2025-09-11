import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _setup_features_stub() -> None:
    """Prepare a lightweight ``features`` package for importing modules."""
    features_pkg = types.ModuleType("features")
    features_pkg.__path__ = [str(ROOT / "features")]
    features_pkg.validate_module = lambda func: func
    sys.modules["features"] = features_pkg


def _stub_indicators() -> None:
    mod = types.ModuleType("indicators")
    mod.atr = lambda h, l, c, n: 0
    mod.bollinger = lambda s, n: (0, s, s)
    mod.rsi = lambda s, n: 0
    mod.sma = lambda s, n: s
    sys.modules["indicators"] = mod


def _import_feature_module(name: str):
    spec = importlib.util.spec_from_file_location(
        f"features.{name}", ROOT / "features" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    sys.modules[f"features.{name}"] = module
    return module


# Set up stub package before importing validators
_setup_features_stub()
from features.validators import assert_no_nan, require_columns  # type: ignore


def test_require_columns_failure() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError) as exc:
        require_columns(df, ["a", "b"])
    assert "b" in str(exc.value)


def test_assert_no_nan_failure() -> None:
    df = pd.DataFrame({"a": [1, None]})
    with pytest.raises(ValueError) as exc:
        assert_no_nan(df)
    assert "a" in str(exc.value)


def test_price_compute_missing_column_raises() -> None:
    _setup_features_stub()
    _stub_indicators()
    price = _import_feature_module("price")
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024", periods=1, freq="T"),
            "Bid": [1.0],
        }
    )
    with pytest.raises(ValueError) as exc:
        price.compute(df)
    assert "Ask" in str(exc.value)


def test_price_compute_nan_raises() -> None:
    _setup_features_stub()
    _stub_indicators()
    price = _import_feature_module("price")
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024", periods=2, freq="T"),
            "Bid": [1.0, 1.1],
            "Ask": [1.2, float("nan")],
        }
    )
    with pytest.raises(ValueError) as exc:
        price.compute(df)
    assert "Ask" in str(exc.value)
    assert "NaN" in str(exc.value)
