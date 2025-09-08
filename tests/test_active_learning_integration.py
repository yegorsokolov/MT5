import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from analysis.active_learning import ActiveLearningQueue, merge_labels


def test_queue_and_label_ingestion(tmp_path):
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    train_idx = np.array([0, 1, 2, 3])
    val_idx = np.array([4, 5])
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = LogisticRegression().fit(X_train, y_train)
    base_acc = accuracy_score(y_val, model.predict(X_val))

    queue = ActiveLearningQueue(
        queue_path=tmp_path / "queue.json", labeled_path=tmp_path / "labels.json"
    )
    val_probs = model.predict_proba(X_val)
    queue.push(val_idx, val_probs, k=1)
    qdata = json.loads((tmp_path / "queue.json").read_text())
    assert len(qdata) == 1
    queued_id = qdata[0]["id"]

    pd.DataFrame({"id": [queued_id], "label": [int(y[queued_id])]}).to_json(
        tmp_path / "labels.json", orient="records"
    )
    labeled = queue.pop_labeled()

    df = pd.DataFrame(X, columns=["x"])
    df["label"] = y
    df.loc[queued_id, "label"] = -1
    df = merge_labels(df, labeled, "label")
    assert df.loc[queued_id, "label"] == y[queued_id]

    model2 = LogisticRegression().fit(df[["x"]].values, df["label"].values)
    new_acc = accuracy_score(y_val, model2.predict(X_val))
    assert new_acc >= base_acc
