import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from mt5.generate_signals import compute_regression_estimates
from models.multi_task_heads import MultiTaskHeadEstimator


@pytest.mark.parametrize("n_samples", [64])
def test_multi_head_regression_predictions(n_samples):
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "f1": rng.normal(size=n_samples),
            "f2": rng.normal(size=n_samples),
            "market_regime": rng.integers(0, 2, size=n_samples),
        }
    )
    targets = pd.DataFrame(
        {
            "direction_1": (features["f1"] + features["f2"] > 0).astype(int),
            "abs_return_1": np.abs(features["f1"]) * 0.01 + 0.001,
            "volatility_1": np.abs(features["f2"]) * 0.02 + 0.002,
        }
    )

    estimator = MultiTaskHeadEstimator(
        classification_targets=["direction_1"],
        abs_targets=["abs_return_1"],
        volatility_targets=["volatility_1"],
        epochs=30,
        learning_rate=0.05,
        random_state=0,
    )
    pipe = Pipeline([("scale", StandardScaler()), ("multi_task", estimator)])
    pipe.fit(features, targets)
    pipe.regression_heads_ = {
        "abs_return": {"type": "multi_task", "columns": ["abs_return_1"]},
        "volatility": {"type": "multi_task", "columns": ["volatility_1"]},
    }
    pipe.regression_feature_columns_ = list(features.columns)
    pipe.regression_trunk_ = ["scale"]

    outputs = compute_regression_estimates([pipe], features, list(features.columns))
    assert set(outputs) >= {"abs_return", "volatility"}

    for head, arr in outputs.items():
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (n_samples,)
        assert np.all(np.isfinite(arr))

    probs = pipe.predict_proba(features)
    assert np.asarray(probs).shape[0] == n_samples
