import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.curriculum import CurriculumScheduler, CurriculumStage


def _make_dataset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X.sum(axis=1) + rng.normal(scale=0.1, size=200)
    return X, y


def _stage_fn(num_features, data):
    X, y = data
    X_train, X_val = X[:150, :num_features], X[150:, :num_features]
    y_train, y_val = y[:150], y[150:]
    # closed-form linear regression using pseudo-inverse
    beta = np.linalg.pinv(X_train) @ y_train
    pred = X_val @ beta
    ss_res = np.sum((y_val - pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    return float(1 - ss_res / ss_tot)


def test_curriculum_progression():
    data = _make_dataset()
    stages = [
        CurriculumStage("single", lambda d=data: _stage_fn(1, d), threshold=0.01),
        CurriculumStage("double", lambda d=data: _stage_fn(2, d), threshold=0.2),
        CurriculumStage("triple", lambda d=data: _stage_fn(3, d), threshold=0.8),
    ]
    scheduler = CurriculumScheduler(stages)
    metrics = scheduler.run()
    # Ensure all stages were completed
    assert scheduler.current_stage == 3
    # Metrics should be strictly improving
    assert metrics[0][1] < metrics[-1][1]
    # Final metric should exceed final threshold
    assert scheduler.final_metric >= stages[-1].threshold
