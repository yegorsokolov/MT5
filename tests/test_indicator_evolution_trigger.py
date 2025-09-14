import json
import ast
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "train.py").read_text()
module = ast.parse(src)
func = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_maybe_evolve_on_degradation")
code = compile(ast.Module(body=[func], type_ignores=[]), "extracted", "exec")
ns = {"Path": Path, "logger": logging.getLogger("test"), "pd": pd}
exec(code, ns)
_maybe_evolve_on_degradation = ns["_maybe_evolve_on_degradation"]

import types, sys
sys.path.append(str(ROOT))
gplearn = types.ModuleType("gplearn")
genetic = types.ModuleType("genetic")
class SymbolicTransformer:  # pragma: no cover - placeholder
    def __init__(self, *a, **k):
        self._best_programs = []
    def fit(self, X, y):
        pass
genetic.SymbolicTransformer = SymbolicTransformer
functions = types.ModuleType("functions")
def make_function(func=None, name=None, arity=None):  # pragma: no cover
    return func
functions.make_function = make_function
gplearn.genetic = genetic
gplearn.functions = functions
sys.modules['gplearn'] = gplearn
sys.modules['gplearn.genetic'] = genetic
sys.modules['gplearn.functions'] = functions

from analysis.indicator_evolution import EvolvedIndicator


def test_evolution_triggers_on_drop(tmp_path, monkeypatch):
    calls = {}

    def fake_evolve(X, y, path):
        path.write_text(json.dumps([{ "name": "evo", "formula": "df.x", "score": 1.0 }]))
        calls["called"] = True
        return [EvolvedIndicator("evo", "df.x", 1.0)]

    monkeypatch.setattr("analysis.indicator_evolution.evolve", fake_evolve)

    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 1.0, 0.0])

    out = _maybe_evolve_on_degradation(
        metric=0.4,
        baseline=0.9,
        X=X,
        y=y,
        threshold=0.1,
        path=tmp_path,
    )
    assert out is True
    assert calls
    file = tmp_path / "evolved_indicators_v1.json"
    data = json.loads(file.read_text())
    assert data[0]["name"] == "evo"


def test_no_evolution_when_metric_ok(tmp_path, monkeypatch):
    called = {}

    def fake_evolve(X, y, path):  # pragma: no cover - should not run
        called["called"] = True
        return []

    monkeypatch.setattr("analysis.indicator_evolution.evolve", fake_evolve)

    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 1.0, 0.0])

    out = _maybe_evolve_on_degradation(
        metric=0.85,
        baseline=0.9,
        X=X,
        y=y,
        threshold=0.1,
        path=tmp_path,
    )
    assert out is False
    assert not called
    assert not list(tmp_path.glob("evolved_indicators_v*.json"))


def test_evolution_discards_when_not_better(tmp_path, monkeypatch):
    def fake_evolve(X, y, path):
        path.write_text(json.dumps([{ "name": "evo", "formula": "df.x", "score": 0.1 }]))
        return [EvolvedIndicator("evo", "df.x", 0.1)]

    monkeypatch.setattr("analysis.indicator_evolution.evolve", fake_evolve)

    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 1.0, 0.0])

    out = _maybe_evolve_on_degradation(
        metric=0.4,
        baseline=0.9,
        X=X,
        y=y,
        threshold=0.1,
        path=tmp_path,
    )
    assert out is False
    assert not list(tmp_path.glob("evolved_indicators_v*.json"))
