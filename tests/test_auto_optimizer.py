import pandas as pd
import sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sklearn_cluster = types.SimpleNamespace(KMeans=object)
sys.modules.setdefault("sklearn", types.SimpleNamespace(cluster=sklearn_cluster))
sys.modules.setdefault("sklearn.cluster", sklearn_cluster)

model_store_stub = types.SimpleNamespace(save_model=lambda *a, **k: "v1")
hot_reload_stub = types.SimpleNamespace(hot_reload=lambda params: None)
models_stub = types.SimpleNamespace(model_store=model_store_stub, hot_reload=hot_reload_stub)
sys.modules.setdefault("models", models_stub)
sys.modules.setdefault("models.model_store", model_store_stub)
sys.modules.setdefault("models.hot_reload", hot_reload_stub)

from tuning.auto_optimizer import AutoOptimizer
import tuning.auto_optimizer as auto


def test_optimizer_triggers_and_improves(monkeypatch):
    regime = [0]

    def fake_detect(df):
        return pd.Series([regime[0]], index=df.index)

    monkeypatch.setattr(auto.regime_detection, "detect_regimes", fake_detect)

    objective_values = [1.0, 2.0, 3.0]

    class DummyTrial:
        def suggest_float(self, name, low, high):
            return 0.1

    class DummyStudy:
        def __init__(self):
            self.enqueued = []
            self.best_params = {}
            self.best_value = 0.0

        def enqueue_trial(self, params):
            self.enqueued.append(params)

        def optimize(self, func, n_trials):
            self.best_value = func(DummyTrial())
            self.best_params = {"lr": 0.1}

    studies = []

    def create_study(**kwargs):
        st = DummyStudy()
        studies.append(st)
        return st

    monkeypatch.setattr(auto.optuna, "create_study", create_study)

    saves = []
    monkeypatch.setattr(
        auto.model_store,
        "save_model",
        lambda model, cfg, perf: saves.append(perf) or "v1",
    )

    hot_calls = []
    monkeypatch.setattr(auto, "hot_reload", lambda params: hot_calls.append(params))

    times = [0.0]
    monkeypatch.setattr(auto.time, "time", lambda: times[0])

    def objective(params, data):
        return objective_values.pop(0)

    opt = AutoOptimizer(objective, n_trials=1, interval=10)
    data = pd.DataFrame({"return": [0.0], "volatility_30": [0.0]})

    opt.maybe_optimize(data)
    assert len(hot_calls) == 1
    assert saves[0]["regime"] == 0
    assert studies[0].enqueued == []
    assert opt.best_score == 1.0

    opt.maybe_optimize(data)
    assert len(hot_calls) == 1

    regime[0] = 1
    opt.maybe_optimize(data)
    assert len(hot_calls) == 2
    assert saves[-1]["regime"] == 1
    assert studies[1].enqueued == [{"lr": 0.1}]
    assert opt.best_score == 2.0

    opt.maybe_optimize(data)
    assert len(hot_calls) == 2

    times[0] = 20
    opt.maybe_optimize(data)
    assert len(hot_calls) == 3
    assert opt.best_score == 3.0
