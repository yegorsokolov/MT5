import importlib.util
import types
import sys
from pathlib import Path
from functools import wraps

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def _setup_features_stub():
    features_pkg = types.ModuleType("features")
    features_pkg.__path__ = [str(ROOT / "features")]

    def validator(suite_name: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                from data.expectations import validate_dataframe
                result = func(*args, **kwargs)
                validate_dataframe(result, suite_name)
                return result

            return wrapper

        return decorator

    features_pkg.validator = validator
    sys.modules["features"] = features_pkg


def _stub_utils():
    util_mod = types.ModuleType("utils")
    util_mod.load_config = lambda: {}
    sys.modules["utils"] = util_mod

    res_mod = types.ModuleType("utils.resource_monitor")
    class RC:
        def __init__(self, cpus=0, memory_gb=0.0, has_gpu=False, gpu_count=0):
            self.cpus = cpus
            self.memory_gb = memory_gb
            self.has_gpu = has_gpu
            self.gpu_count = gpu_count
    res_mod.ResourceCapabilities = RC
    res_mod.monitor = types.SimpleNamespace(capabilities=RC(), subscribe=lambda: types.SimpleNamespace())
    sys.modules["utils.resource_monitor"] = res_mod


def _stub_expectations(called):
    stub = types.ModuleType("data.expectations")
    def _validate(df, suite, **_):
        called["suite"] = suite
        raise ValueError("failed")
    stub.validate_dataframe = _validate
    sys.modules["data.expectations"] = stub
    data_pkg = sys.modules.setdefault("data", types.ModuleType("data"))
    data_pkg.expectations = stub


def test_order_flow_validation_failure():
    _setup_features_stub()
    _stub_utils()
    called = {}
    _stub_expectations(called)

    spec = importlib.util.spec_from_file_location("features.order_flow", ROOT / "features" / "order_flow.py")
    order_flow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(order_flow)
    sys.modules["features.order_flow"] = order_flow

    df = pd.DataFrame({"bid_sz_0": [1.0], "ask_sz_0": [2.0]})
    with pytest.raises(ValueError):
        order_flow.compute(df)
    assert called["suite"] == "order_flow"


def test_cross_asset_validation_failure():
    _setup_features_stub()
    _stub_utils()
    sys.modules["data.graph_builder"] = types.SimpleNamespace(build_rolling_adjacency=lambda df: {})
    data_pkg = sys.modules.setdefault("data", types.ModuleType("data"))
    data_pkg.graph_builder = sys.modules["data.graph_builder"]
    called = {}
    _stub_expectations(called)

    spec = importlib.util.spec_from_file_location("features.cross_asset", ROOT / "features" / "cross_asset.py")
    cross_asset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cross_asset)
    sys.modules["features.cross_asset"] = cross_asset

    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020", periods=3, freq="D").tolist() * 2,
        "Symbol": ["A", "B", "A", "B", "A", "B"],
        "return": [0.1, 0.2, -0.1, 0.05, 0.0, -0.05],
    })

    with pytest.raises(ValueError):
        cross_asset.compute(df)
    assert called["suite"] == "cross_asset"
