import pytest
import sys
import types
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load backtest module with dummy logging to avoid import side effects
spec = importlib.util.spec_from_file_location("backtest", Path(__file__).resolve().parents[1] / "backtest.py")
sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=object)
backtest = importlib.util.module_from_spec(spec)
backtest.setup_logging = lambda: None
sys.modules['log_utils'] = types.SimpleNamespace(setup_logging=lambda: None, log_exceptions=lambda f: f)
spec.loader.exec_module(backtest)

trailing_stop = backtest.trailing_stop


def test_trailing_stop_tightens():
    stop = trailing_stop(1.0, 1.05, 0.99, 0.01)
    assert stop == pytest.approx(1.04)
    stop = trailing_stop(1.0, 1.07, stop, 0.01)
    assert stop == pytest.approx(1.06)


def test_trailing_stop_does_not_loosen():
    stop = 1.04
    new_stop = trailing_stop(1.0, 1.05, stop, 0.02)
    assert new_stop == stop
