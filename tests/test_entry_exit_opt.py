import numpy as np
import pandas as pd
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _DummyHist:
    def record(self, *_, **__):
        pass


class _DummyMeter:
    def create_histogram(self, *_, **__):
        return _DummyHist()


sys.modules["telemetry"] = types.SimpleNamespace(
    get_meter=lambda name=None: _DummyMeter(), get_tracer=lambda name=None: None
)

from tuning.entry_exit_opt import EntryExitOptimizer


def test_entry_exit_optimizer_triggers_and_persists(monkeypatch):
    np.random.seed(0)
    data = pd.DataFrame({"return": np.random.normal(0, 0.01, size=300)})

    regimes = iter([0, 0, 1, 1])

    def fake_detect(df):
        r = next(regimes)
        return pd.Series([r] * len(df))

    monkeypatch.setattr(
        EntryExitOptimizer, "_current_regime", lambda self, d: int(fake_detect(d).iloc[-1])
    )

    class DummyTrial:
        def __init__(self, params):
            self.params = params

        def suggest_float(self, name, low, high):
            return self.params[name]

        def suggest_int(self, name, low, high):
            return self.params[name]

    trial_params = iter(
        [
            {"stop": -0.01, "profit": 0.02, "holding": 5},
            {"stop": -0.01, "profit": 0.02, "holding": 5},
            {"stop": -0.01, "profit": 0.03, "holding": 5},
        ]
    )

    class DummyStudy:
        def __init__(self):
            self.best_params = {}
            self.best_value = 0.0
            self.p = next(trial_params)

        def enqueue_trial(self, params):
            pass

        def optimize(self, func, n_trials, **kwargs):
            trial = DummyTrial(self.p)
            self.best_value = func(trial)
            self.best_params = trial.params

    import tuning.entry_exit_opt as module
    monkeypatch.setattr(module.optuna, "create_study", lambda **_: DummyStudy())
    monkeypatch.setattr(EntryExitOptimizer, "_objective", lambda self, p, d: p["profit"])

    saved = []

    def fake_save_model(model, cfg, perf, **kwargs):
        saved.append(perf["regime"])
        return "id"

    monkeypatch.setattr(module.model_store, "save_model", fake_save_model)

    hot = []

    def fake_hot_reload(params):
        hot.append(params)

    monkeypatch.setattr(module, "hot_reload", fake_hot_reload)

    times = iter([0, 0, 5, 6, 6, 20, 20])
    monkeypatch.setattr(module.time, "time", lambda: next(times))

    opt = EntryExitOptimizer(n_trials=1, interval=10)

    res1 = opt.maybe_optimize(data)
    assert res1 == {"stop": -0.01, "profit": 0.02, "holding": 5}
    assert hot[-1] == res1
    assert saved[-1] == 0

    res2 = opt.maybe_optimize(data)
    assert res2 is None

    res3 = opt.maybe_optimize(data)
    assert res3 == {"stop": -0.01, "profit": 0.02, "holding": 5}
    assert saved[-1] == 1
    assert set(opt.best_params) == {0, 1}

    res4 = opt.maybe_optimize(data)
    assert res4 == {"stop": -0.01, "profit": 0.03, "holding": 5}
    assert len(hot) == 3
