import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "train_ensemble", ROOT / "train_ensemble.py"
)
_te = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["train_ensemble"] = _te
_spec.loader.exec_module(_te)  # type: ignore

ResourceCapabilities = _te.ResourceCapabilities
train_moe_ensemble = _te.train_moe_ensemble


def test_mixture_beats_individual_experts() -> None:
    """Mixture-of-experts should outperform any single expert on regime data."""

    data_trend = np.arange(1, 11, dtype=float)
    data_mean = 10 * (0.5) ** np.arange(10)
    data_macro = np.zeros(10)
    data = np.concatenate([data_trend, data_mean, data_macro])

    caps = ResourceCapabilities(4, 16, False, gpu_count=0)

    histories, regimes, targets = [], [], []
    for t in range(2, len(data)):
        histories.append(data[t - 2 : t])
        regimes.append(0 if t < 12 else 1 if t < 22 else 2)
        targets.append(data[t])

    mix_pred, expert_preds = train_moe_ensemble(histories, regimes, targets, caps)
    targets_arr = np.array(targets)

    def mse(arr: np.ndarray) -> float:
        return float(np.mean((arr - targets_arr) ** 2))

    mix_err = mse(mix_pred)
    for i in range(expert_preds.shape[1]):
        assert mix_err < mse(expert_preds[:, i])
