import sys
import types
from pathlib import Path

import pytest


@pytest.mark.skip(reason="requires heavy optional dependencies")
def test_generate_signals_closed_market(monkeypatch):
    # Ensure repository root on path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    calls = {}

    class DummyAlgo:
        @classmethod
        def load(cls, *args, **kwargs):
            return cls()

    # Stub out heavy optional dependencies before importing module
    sys.modules['stable_baselines3'] = types.SimpleNamespace(PPO=DummyAlgo, SAC=DummyAlgo, A2C=DummyAlgo)
    sb3_contrib = types.ModuleType("sb3_contrib")
    sb3_contrib.TRPO = DummyAlgo
    sb3_contrib.RecurrentPPO = DummyAlgo
    sb3_contrib.qrdqn = types.SimpleNamespace(QRDQN=DummyAlgo)
    sys.modules['sb3_contrib'] = sb3_contrib
    sys.modules['sb3_contrib.qrdqn'] = sb3_contrib.qrdqn
    sys.modules['river'] = types.SimpleNamespace(compose=types.SimpleNamespace())
    sys.modules['train_rl'] = types.SimpleNamespace(
        TradingEnv=object, DiscreteTradingEnv=object, RLLibTradingEnv=object, HierarchicalTradingEnv=object
    )
    sys.modules['signal_queue'] = types.SimpleNamespace(
        get_async_publisher=lambda *a, **k: None,
        publish_dataframe_async=lambda *a, **k: None,
    )

    def fake_backtest(cfg):
        calls['called'] = True

    sys.modules['backtest'] = types.SimpleNamespace(run_rolling_backtest=fake_backtest)

    import generate_signals

    monkeypatch.setattr(generate_signals, 'is_market_open', lambda: False)
    monkeypatch.setattr(generate_signals, 'load_config', lambda: {})

    def stop(*args, **kwargs):
        raise SystemExit

    monkeypatch.setattr(generate_signals, 'load_models', stop)
    monkeypatch.setattr(sys, 'argv', ['generate_signals.py'])

    with pytest.raises(SystemExit):
        generate_signals.main()

    assert calls.get('called')
