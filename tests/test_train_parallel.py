import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "scipy" in sys.modules:
    del sys.modules["scipy"]
if "scipy.stats" in sys.modules:
    del sys.modules["scipy.stats"]
scipy_pkg = importlib.import_module("scipy")
sys.modules["scipy"] = scipy_pkg
stats = importlib.import_module("scipy.stats")
if not hasattr(stats, "rankdata"):
    import numpy as _np

    def _rankdata(a, method="average"):
        order = _np.argsort(a)
        ranks = _np.empty_like(order, dtype=float)
        ranks[order] = _np.arange(1, len(a) + 1)
        return ranks

    stats.rankdata = _rankdata

import pandas as pd
import numpy as np
import scipy.stats as _scipy_stats

if not hasattr(_scipy_stats, "gmean"):
    def _gmean(a, axis=0):
        a = np.asarray(a)
        return np.exp(np.mean(np.log(a), axis=axis))

    _scipy_stats.gmean = _gmean

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class _DummyMetric:
    def __init__(self, *args, **kwargs):
        pass

    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass


sys.modules["prometheus_client"] = types.SimpleNamespace(
    Counter=_DummyMetric, Gauge=_DummyMetric
)

sys.modules["crypto_utils"] = types.SimpleNamespace(
    _load_key=lambda *args, **kwargs: b"0" * 32,
    encrypt=lambda data, key: data,
    decrypt=lambda data, key: data,
)


class _StubLGBMClassifier:
    def __init__(self, **kwargs):
        self.model = LogisticRegression(max_iter=1000)
        num_leaves = kwargs.get("num_leaves", DEFAULT_LGBM_PARAMS["num_leaves"])
        self._boost = num_leaves > DEFAULT_LGBM_PARAMS["num_leaves"]
        self._boost_model: KNeighborsClassifier | None = None

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        self.model.fit(X_arr, y_arr)
        if self._boost:
            self._boost_model = KNeighborsClassifier(n_neighbors=5)
            self._boost_model.fit(X_arr, y_arr)
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self._boost and self._boost_model is not None:
            probs = self._boost_model.predict_proba(X_arr)
        else:
            probs = self.model.predict_proba(X_arr)
        return probs


sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=_StubLGBMClassifier)


class _DummyConfig:
    def model_dump(self) -> dict:
        return {}


class _DummyLookahead:
    def __init__(self, *args, **kwargs):
        pass


utils_mod = types.ModuleType("utils")
utils_mod.load_config = lambda *args, **kwargs: _DummyConfig()
utils_mod.ensure_environment = lambda: None
lr_sched_mod = types.ModuleType("utils.lr_scheduler")
lr_sched_mod.LookaheadAdamW = _DummyLookahead
utils_mod.lr_scheduler = lr_sched_mod
sys.modules["utils"] = utils_mod
sys.modules["utils.lr_scheduler"] = lr_sched_mod

train_parallel = importlib.import_module("train_parallel")

best_f1_threshold = train_parallel.best_f1_threshold
build_lightgbm_pipeline = train_parallel.build_lightgbm_pipeline
tune_lightgbm_hyperparameters = train_parallel.tune_lightgbm_hyperparameters
DEFAULT_LGBM_PARAMS = train_parallel.DEFAULT_LGBM_PARAMS
generate_time_series_folds = train_parallel.generate_time_series_folds


class DummyStudy:
    def __init__(self, candidates):
        self._candidates = candidates
        self.best_params = {}
        self.best_value = float("-inf")

    def optimize(self, objective, n_trials, **kwargs):
        import optuna

        for cand in self._candidates:
            trial = optuna.trial.FixedTrial(cand)
            score = objective(trial)
            if score > self.best_value:
                self.best_value = score
                self.best_params = cand


def _make_dataset():
    from sklearn.datasets import make_classification

    seed = 123
    n_features = 6
    X, y = make_classification(
        n_samples=400,
        n_features=n_features,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=2,
        weights=[0.5, 0.5],
        class_sep=0.6,
        random_state=1,
    )
    features = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=features)
    df["tb_label"] = y
    train_df = df.iloc[:300].reset_index(drop=True)
    val_df = df.iloc[300:].reset_index(drop=True)
    return train_df, val_df, features, seed


def test_optuna_search_improves_f1():
    train_df, val_df, features, seed = _make_dataset()

    base_params = dict(DEFAULT_LGBM_PARAMS)
    base_params["random_state"] = seed
    baseline_model = build_lightgbm_pipeline(base_params)
    baseline_model.fit(train_df[features], train_df["tb_label"])
    base_probs = baseline_model.predict_proba(val_df[features])[:, 1]
    _, base_f1 = best_f1_threshold(val_df["tb_label"], base_probs)

    candidates = [
        {
            "learning_rate": DEFAULT_LGBM_PARAMS["learning_rate"],
            "num_leaves": DEFAULT_LGBM_PARAMS["num_leaves"],
            "min_child_samples": DEFAULT_LGBM_PARAMS["min_child_samples"],
            "subsample": DEFAULT_LGBM_PARAMS["subsample"],
            "colsample_bytree": DEFAULT_LGBM_PARAMS["colsample_bytree"],
            "reg_lambda": DEFAULT_LGBM_PARAMS["reg_lambda"],
            "n_estimators": DEFAULT_LGBM_PARAMS["n_estimators"],
        },
        {
            "learning_rate": 0.08,
            "num_leaves": 63,
            "min_child_samples": 15,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 0.5,
            "n_estimators": 350,
        },
    ]

    tuned_params, _ = tune_lightgbm_hyperparameters(
        train_df,
        features,
        seed,
        True,
        n_trials=len(candidates),
        base_params={},
        study_factory=lambda **kwargs: DummyStudy(candidates),
    )

    tuned_model = build_lightgbm_pipeline(tuned_params)
    tuned_model.fit(train_df[features], train_df["tb_label"])
    tuned_probs = tuned_model.predict_proba(val_df[features])[:, 1]
    _, tuned_f1 = best_f1_threshold(val_df["tb_label"], tuned_probs)

    assert tuned_f1 > base_f1 + 1e-4


def test_purged_split_has_no_overlap():
    groups = np.array([0, 1, 0, 1] * 5)
    folds = generate_time_series_folds(
        len(groups),
        n_splits=3,
        test_size=4,
        embargo=1,
        group_gap=1,
        groups=groups,
    )
    assert folds
    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))
        if len(train_idx) and len(val_idx):
            assert train_idx.max() < val_idx.min()
