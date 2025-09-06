import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))


def train_simple(X, y, use_pseudo=False, pseudo_dir=None):
    from sklearn.linear_model import LogisticRegression

    if use_pseudo and pseudo_dir is not None:
        p_parquet = Path(pseudo_dir) / "pseudo_labels.parquet"
        p_csv = Path(pseudo_dir) / "pseudo_labels.csv"
        if p_parquet.exists():
            df_p = pd.read_parquet(p_parquet)
        elif p_csv.exists():
            df_p = pd.read_csv(p_csv)
        else:
            df_p = None
        if df_p is not None:
            X = np.vstack([X, df_p.drop(columns=["pseudo_label"]).values])
            y = np.concatenate([y, df_p["pseudo_label"].values])
    clf = LogisticRegression().fit(X, y)
    return clf


def test_pseudo_label_improves(tmp_path):
    import sys
    sys.modules.pop("scipy", None)
    sys.modules.pop("scipy.stats", None)
    from analysis.pseudo_labeler import generate_pseudo_labels
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = X[:10], X[10:]
    y_train, y_test = y[:10], y[10:]

    model = train_simple(X_train, y_train)
    base_acc = model.score(X_test, y_test)

    df_unlabeled = pd.DataFrame(X[10:], columns=["f1", "f2"])
    generate_pseudo_labels(
        model,
        df_unlabeled,
        y_true=y[10:],
        threshold=0.5,
        output_dir=tmp_path,
        report_dir=tmp_path / "report",
    )
    model_pl = train_simple(X_train, y_train, use_pseudo=True, pseudo_dir=tmp_path)
    pl_acc = model_pl.score(X_test, y_test)
    assert pl_acc >= base_acc

    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    assert set(metrics) == {"precision", "recall"}
