import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
import scipy  # ensure real scipy is loaded
from analysis.feature_evolver import FeatureEvolver


class _Prog:
    def __init__(self, values=None):
        self.values = values

    def execute(self, X):
        return self.values

    def __str__(self):  # pragma: no cover - trivial
        return "X0 + X1"


def _patch_store(tmp_path):
    import feature_store as fs
    import analysis.feature_evolver as fe

    fs.STORE_DIR = tmp_path / "fs"
    fs.INDEX_FILE = fs.STORE_DIR / "index.json"

    class _DummyST:
        def __init__(self, generations=0, population_size=0, hall_of_fame=0, n_components=1, random_state=0, n_jobs=1):
            self.n_components = n_components
            self._best_programs = [_Prog() for _ in range(n_components)]

        def fit(self, X, y):
            vals = X[:, 0] + X[:, 1]
            for prog in self._best_programs:
                prog.values = vals

        def transform(self, X):
            vals = X[:, 0] + X[:, 1]
            return np.column_stack([vals for _ in range(self.n_components)])

    fe.SymbolicTransformer = _DummyST  # type: ignore

    class _LR:
        def fit(self, X, y):  # pragma: no cover - simple stub
            pass

    fe.LinearRegression = _LR  # type: ignore

    fe.cross_val_score = lambda *a, **k: np.array([0.0])  # type: ignore

    class _KF:
        def __init__(self, *a, **k):
            pass

    fe.KFold = _KF  # type: ignore


def test_feature_evolver(tmp_path):
    _patch_store(tmp_path)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
    df["return"] = df["a"] * df["b"] + rng.normal(scale=0.1, size=80)
    df["market_regime"] = 1

    store = tmp_path / "store"
    module_path = tmp_path / "features_mod.py"
    evolver = FeatureEvolver(store_dir=store)

    df2 = evolver.maybe_evolve(
        df.copy(),
        target_col="return",
        module_path=module_path,
        generations=5,
        population_size=100,
    )
    gp_cols = [c for c in df2.columns if c.startswith("gp_feat_")]
    assert gp_cols, "no evolved features created"

    # manifest should log expression and score
    manifest_text = (store / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    assert manifest[0]["expression"]
    assert isinstance(manifest[0]["score"], float)

    # running again should yield identical manifest and not duplicate file output
    file_text = module_path.read_text()
    evolver.maybe_evolve(df.copy(), target_col="return", module_path=module_path)
    assert manifest_text == (store / "manifest.json").read_text()
    assert file_text == module_path.read_text()

    # ensure we can load and apply stored features on new data
    df_new = evolver.apply_stored_features(df.copy())
    for col in gp_cols:
        assert col in df_new.columns


def test_regime_change_triggers_evolution(tmp_path):
    _patch_store(tmp_path)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
    df["return"] = df["a"] * df["b"] + rng.normal(scale=0.1, size=80)
    df["market_regime"] = 1

    evolver = FeatureEvolver(store_dir=tmp_path / "store2")

    df1 = evolver.maybe_evolve(
        df.copy(),
        target_col="return",
        generations=5,
        population_size=100,
    )
    initial_cols = [c for c in df1.columns if c.startswith("gp_feat_")]
    assert initial_cols, "no evolved features created"

    df_same = evolver.apply_stored_features(df.copy())
    df_same = evolver.maybe_evolve(df_same, target_col="return")
    assert [c for c in df_same.columns if c.startswith("gp_feat_")] == initial_cols

    df2 = df.copy()
    df2["market_regime"] = 2
    df2 = evolver.apply_stored_features(df2)
    df2 = evolver.maybe_evolve(df2, target_col="return", generations=5, population_size=100)
    cols2 = [c for c in df2.columns if c.startswith("gp_feat_")]
    assert len(cols2) > len(initial_cols)

    from feature_store import list_versions

    versions = list_versions()
    for col in cols2:
        assert col in versions
