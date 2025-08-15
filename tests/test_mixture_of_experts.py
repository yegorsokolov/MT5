from dataclasses import dataclass

import importlib.util
from pathlib import Path

import numpy as np

from model_registry import ModelRegistry, ResourceCapabilities

_spec = importlib.util.spec_from_file_location(
    "mixture_of_experts", Path(__file__).resolve().parents[1] / "models" / "mixture_of_experts.py"
)
_moe = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_moe)  # type: ignore
TrendExpert = _moe.TrendExpert
MeanReversionExpert = _moe.MeanReversionExpert
MacroExpert = _moe.MacroExpert


@dataclass
class DummyMonitor:
    capabilities: ResourceCapabilities

    def start(self) -> None:  # pragma: no cover - no background tasks in tests
        pass


def test_smooth_switching() -> None:
    monitor = DummyMonitor(ResourceCapabilities(4, 16, False, gpu_count=0))
    registry = ModelRegistry(monitor, auto_refresh=False)
    w0 = registry.moe.weights(0.0, monitor.capabilities)
    w_half = registry.moe.weights(0.5, monitor.capabilities)
    w1 = registry.moe.weights(1.0, monitor.capabilities)
    assert w0[0] > 0.8 and w0[1] < 0.15
    assert w1[1] > 0.8 and w1[0] < 0.15
    assert abs(w_half[0] - w_half[1]) < 0.2


def test_mixed_regime_accuracy() -> None:
    monitor = DummyMonitor(ResourceCapabilities(4, 16, False, gpu_count=0))
    registry = ModelRegistry(monitor, auto_refresh=False)
    data_trend = np.arange(1, 31, dtype=float)
    data_mean = 10 * (0.5) ** np.arange(30)
    data_macro = np.zeros(30)
    data = np.concatenate([data_trend, data_mean, data_macro])
    trend_model = TrendExpert()
    mean_model = MeanReversionExpert()
    macro_model = MacroExpert()
    preds_mix, preds_trend, preds_mean, preds_macro, targets = [], [], [], [], []
    for t in range(2, len(data)):
        history = data[t - 2 : t]
        regime = 0 if t < 30 else 1 if t < 60 else 2
        target = data[t]
        preds_mix.append(registry.predict_mixture(history, regime))
        preds_trend.append(trend_model.predict(history))
        preds_mean.append(mean_model.predict(history))
        preds_macro.append(macro_model.predict(history))
        targets.append(target)
    targets_arr = np.array(targets)
    def mse(preds: list[float]) -> float:
        arr = np.array(preds)
        return float(np.mean((arr - targets_arr) ** 2))
    mix_err = mse(preds_mix)
    assert mix_err < mse(preds_trend)
    assert mix_err < mse(preds_mean)
    assert mix_err < mse(preds_macro)
