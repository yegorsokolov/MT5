import numpy as np
import torch
from torch.utils.data import TensorDataset
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.curriculum import build_strategy_curriculum
from models.meta_learner import (
    _LinearModel,
    meta_train_reptile,
    meta_train_maml,
    fine_tune_model,
    _steps_to,
)


def _generate_task(weight, n_samples: int = 40):
    X = np.random.randn(n_samples, len(weight))
    y = (X @ weight > 0).astype(float)
    split = n_samples // 2
    train = TensorDataset(
        torch.tensor(X[:split], dtype=torch.float32),
        torch.tensor(y[:split], dtype=torch.float32),
    )
    val = TensorDataset(
        torch.tensor(X[split:], dtype=torch.float32),
        torch.tensor(y[split:], dtype=torch.float32),
    )
    return train, val


def test_curriculum_and_meta_learning():
    np.random.seed(0)
    torch.manual_seed(0)
    state = {}

    def stage_simple():
        tasks = [_generate_task(np.array([1.0])), _generate_task(np.array([-1.0]))]
        build = lambda: _LinearModel(1)
        state["s1"] = meta_train_reptile(tasks, build, epochs=20)
        _, hist = fine_tune_model(state["s1"], tasks[0][0], build, steps=5, lr=0.5)
        return hist[-1]

    def stage_combo():
        tasks = [_generate_task(np.array([2.0, -2.0])), _generate_task(np.array([-2.0, 2.0]))]
        build = lambda: _LinearModel(2)
        state["s2"] = meta_train_reptile(tasks, build, epochs=25)
        _, hist = fine_tune_model(state["s2"], tasks[0][0], build, steps=5, lr=0.5)
        return hist[-1]

    def stage_graph():
        tasks = [_generate_task(np.array([1.0, 1.0, -1.0])), _generate_task(np.array([-1.0, 1.0, 1.0]))]
        build = lambda: _LinearModel(3)
        state["s3"] = meta_train_maml(tasks, build, epochs=40)
        eval_ds, _ = _generate_task(np.array([1.0, -1.0, 1.0]))
        state["eval"] = eval_ds
        _, hist = fine_tune_model(state["s3"], eval_ds, build, steps=5, lr=0.5)
        return hist[-1]

    sched = build_strategy_curriculum(stage_simple, stage_combo, stage_graph, thresholds=(0.5, 0.6, 0.6))
    sched.run()
    assert sched.current_stage == 3

    build_eval = lambda: _LinearModel(3)
    _, base_hist = fine_tune_model(build_eval().state_dict(), state["eval"], build_eval, steps=5, lr=0.1)
    _, meta_hist = fine_tune_model(state["s3"], state["eval"], build_eval, steps=5, lr=0.1)
    assert meta_hist[-1] >= base_hist[-1] - 0.1
