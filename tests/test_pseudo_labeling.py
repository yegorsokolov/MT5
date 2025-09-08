import json
from pathlib import Path

import numpy as np
import pandas as pd



def train_simple(X, y, use_pseudo=False, pseudo_dir=None):
    from sklearn.linear_model import LogisticRegression

    if use_pseudo and pseudo_dir is not None:
        p_path = Path(pseudo_dir) / "pseudo_labels.csv"
        if p_path.exists():
            df_p = pd.read_csv(p_path)
            X = np.vstack([X, df_p.drop(columns=["pseudo_label"]).values])
            y = np.concatenate([y, df_p["pseudo_label"].values])
    clf = LogisticRegression().fit(X, y)
    return clf


def test_pseudo_labels_persist_and_improve_f1(tmp_path):
    import sys
    from sklearn.metrics import f1_score
    from analysis.pseudo_labeler import generate_pseudo_labels

    sys.modules.pop("scipy", None)
    sys.modules.pop("scipy.stats", None)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = X[:10], X[10:]
    y_train, y_test = y[:10], y[10:]

    model = train_simple(X_train, y_train)
    base_f1 = f1_score(y_test, model.predict(X_test))

    df_unlabeled = pd.DataFrame(X_test, columns=["f1", "f2"])
    generate_pseudo_labels(
        model,
        df_unlabeled,
        y_true=y_test,
        threshold=0.5,
        output_dir=tmp_path,
        report_dir=tmp_path / "report",
    )
    assert (tmp_path / "pseudo_labels.csv").exists()

    model_pl = train_simple(X_train, y_train, use_pseudo=True, pseudo_dir=tmp_path)
    pl_f1 = f1_score(y_test, model_pl.predict(X_test))
    assert pl_f1 >= base_f1

    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    assert set(metrics) == {"precision", "recall"}
