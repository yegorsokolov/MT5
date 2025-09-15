import numpy as np
import pandas as pd
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from analysis.active_learning import ActiveLearningQueue, merge_labels


class SimpleClf:
    def fit(self, X, y):
        X = np.asarray(X).ravel()
        self.threshold = float(X[y == 1].mean()) if np.any(y == 1) else 0.0
        return self

    def predict(self, X):
        X = np.asarray(X).ravel()
        return (X >= self.threshold).astype(int)

    def predict_proba(self, X):
        preds = self.predict(X)
        return np.vstack([1 - preds, preds]).T


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def test_active_learning_improves_validation(tmp_path):
    # Training data with one unlabeled sample (index 3)
    X_train = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_val = np.array([[2.5], [3.5]], dtype=float)
    y_val = np.array([0, 1])

    unlabeled_idx = np.array([3])
    y_missing = y_train.copy()
    y_missing[unlabeled_idx] = -1
    mask = y_missing != -1

    model = SimpleClf().fit(X_train[mask], y_missing[mask])
    base_acc = accuracy(y_val, model.predict(X_val))

    queue = ActiveLearningQueue(
        queue_path=tmp_path / "queue.json", labeled_path=tmp_path / "labels.json"
    )
    probs = np.array([[0.6, 0.4]])
    queue.push_low_confidence(unlabeled_idx, probs, threshold=0.8)

    # Simulate human returning correct label
    pd.DataFrame({"id": unlabeled_idx, "label": y_train[unlabeled_idx]}).to_json(
        tmp_path / "labels.json", orient="records"
    )
    labeled = queue.pop_labeled()

    train_df = pd.DataFrame(X_train, columns=["x"])
    train_df["label"] = y_missing
    train_df = merge_labels(train_df, labeled, "label")

    model2 = SimpleClf().fit(train_df[["x"]].values, train_df["label"].values)
    new_acc = accuracy(y_val, model2.predict(X_val))

    assert new_acc >= base_acc
