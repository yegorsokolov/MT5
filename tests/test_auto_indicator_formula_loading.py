import importlib.util
import types
import sys
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("features")
pkg.__path__ = [str(ROOT / "features")]
sys.modules.setdefault("features", pkg)

spec = importlib.util.spec_from_file_location(
    "features.auto_indicators", ROOT / "features" / "auto_indicators.py"
)
auto_indicators = importlib.util.module_from_spec(spec)
sys.modules["features.auto_indicators"] = auto_indicators
assert spec.loader is not None
spec.loader.exec_module(auto_indicators)


def test_apply_loads_evolved_formulas(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    registry = tmp_path / "auto_indicators.json"
    registry.write_text("[]")
    formula = tmp_path / "evolved_indicators_v1.json"
    formula.write_text(json.dumps([{"name": "x_sq", "formula": "df.x ** 2"}]))

    out = auto_indicators.apply(df, registry_path=registry, formula_dir=tmp_path)
    assert "x_sq" in out.columns
    assert out["x_sq"].tolist() == [1, 4, 9]


def test_generate_loads_evolved_formulas(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    registry = tmp_path / "auto_indicators.json"
    registry.write_text("[]")
    formula = tmp_path / "evolved_indicators_v1.json"
    formula.write_text(json.dumps([{"name": "x_sq", "formula": "df.x ** 2"}]))

    # simple model returning lag and window parameters
    model = lambda inputs: (1, 1)
    out, desc = auto_indicators.generate(
        df,
        model,
        asset_features=[],
        regime=[],
        registry_path=registry,
        formula_dir=tmp_path,
    )
    assert "x_sq" in out.columns
    assert out["x_sq"].tolist() == [1, 4, 9]
    assert desc == {"lag": 1, "window": 1}
