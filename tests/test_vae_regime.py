import sys
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from analysis.regime_detection import periodic_reclassification  # noqa: E402


def _make_data(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 1, size=n)
    vol = np.concatenate([np.full(n // 2, 0.5), np.full(n // 2, 2.0)])
    return pd.DataFrame({"return": ret, "volatility_30": vol})


def test_vae_regime_labels_change_on_shift():
    df = _make_data()
    labeled = periodic_reclassification(df, step=len(df) // 2, n_states=2)
    regimes = labeled["vae_regime"].to_numpy()
    first = regimes[: len(regimes) // 2]
    second = regimes[len(regimes) // 2 :]
    assert np.bincount(first).argmax() != np.bincount(second).argmax()


def test_vae_regime_improves_f1():
    df = _make_data()
    df["target"] = np.array([0] * (len(df) // 2) + [1] * (len(df) // 2))
    labeled = periodic_reclassification(df, step=len(df) // 2, n_states=2)

    X = labeled[["return"]].values
    y = labeled["target"].values
    clf = LogisticRegression().fit(X, y)
    f1_no = f1_score(y, clf.predict(X))

    X_regime = labeled[["return", "vae_regime"]].values
    clf2 = LogisticRegression().fit(X_regime, y)
    f1_with = f1_score(y, clf2.predict(X_regime))

    assert f1_with > f1_no
