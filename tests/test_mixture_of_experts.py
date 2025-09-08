import importlib.util
from pathlib import Path
import sys

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "train_ensemble", Path(__file__).resolve().parents[1] / "train_ensemble.py"
)
_te = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["train_ensemble"] = _te
_spec.loader.exec_module(_te)  # type: ignore

MacroExpert = _te.MacroExpert
MeanReversionExpert = _te.MeanReversionExpert
ResourceCapabilities = _te.ResourceCapabilities
TrendExpert = _te.TrendExpert
predict_mixture = _te.predict_mixture


def test_mixture_beats_individual_experts() -> None:
    """Mixture-of-experts should outperform any single expert on regime data."""

    data_trend = np.arange(1, 11, dtype=float)
    data_mean = 10 * (0.5) ** np.arange(10)
    data_macro = np.zeros(10)
    data = np.concatenate([data_trend, data_mean, data_macro])

    trend_model = TrendExpert()
    mean_model = MeanReversionExpert()
    macro_model = MacroExpert()
    caps = ResourceCapabilities(4, 16, False, gpu_count=0)

    preds_mix, preds_trend, preds_mean, preds_macro, targets = [], [], [], [], []
    for t in range(2, len(data)):
        history = data[t - 2 : t]
        regime = 0 if t < 12 else 1 if t < 22 else 2
        preds_mix.append(predict_mixture(history, regime, caps))
        preds_trend.append(trend_model.predict(history))
        preds_mean.append(mean_model.predict(history))
        preds_macro.append(macro_model.predict(history))
        targets.append(data[t])

    targets_arr = np.array(targets)

    def mse(arr: list[float]) -> float:
        a = np.array(arr)
        return float(np.mean((a - targets_arr) ** 2))

    mix_err = mse(preds_mix)
    assert mix_err < mse(preds_trend)
    assert mix_err < mse(preds_mean)
    assert mix_err < mse(preds_macro)
