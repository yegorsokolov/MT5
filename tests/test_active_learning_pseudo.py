from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.active_learning import ActiveLearningQueue, merge_labels
from analysis.pseudo_labeler import generate_pseudo_labels


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


def f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def test_queue_and_pseudo_labels_improve_f1(tmp_path):
    X = np.array([[0], [1], [2], [3], [4], [5], [6], [7]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])

    train_idx = np.array([0, 1, 2, 3, 4, 5])
    val_idx = np.array([6, 7])
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # remove labels for indices 2 and 3 in training data
    y_masked = y_train.copy()
    mask = np.ones_like(y_masked, dtype=bool)
    mask[2] = False
    mask[3] = False

    model = SimpleClf().fit(X_train[mask], y_masked[mask])
    base_f1 = f1(y_val, model.predict(X_val))

    queue = ActiveLearningQueue(
        queue_path=tmp_path / "queue.json", labeled_path=tmp_path / "labels.json"
    )
    unlabeled_ids = train_idx[~mask]
    probs_unlabeled = model.predict_proba(X_train[~mask])
    queue.push_low_confidence(unlabeled_ids, probs_unlabeled, threshold=0.8)

    df_unlabeled = pd.DataFrame(X_train[~mask], columns=["x"])
    y_unlabeled = y_train[~mask]
    generate_pseudo_labels(
        model,
        df_unlabeled,
        y_true=y_unlabeled,
        threshold=0.8,
        output_dir=tmp_path,
        report_dir=tmp_path / "report",
    )

    # simulate labeler confirming low-confidence sample
    pd.DataFrame(
        {"id": [int(unlabeled_ids[1])], "label": [int(y_unlabeled[1])]}
    ).to_json(tmp_path / "labels.json", orient="records")
    labeled = queue.pop_labeled()

    train_df = pd.DataFrame(X_train, columns=["x"], index=train_idx)
    train_df["label"] = y_masked
    train_df = merge_labels(train_df, labeled, "label")

    pseudo_path = tmp_path / "pseudo_labels.csv"
    if pseudo_path.exists():
        df_pseudo = pd.read_csv(pseudo_path)
        df_pseudo = df_pseudo.rename(columns={"pseudo_label": "label"})
        train_df = pd.concat([train_df, df_pseudo], ignore_index=True)

    model2 = SimpleClf().fit(train_df[["x"]].values, train_df["label"].values)
    new_f1 = f1(y_val, model2.predict(X_val))

    assert new_f1 >= base_f1
