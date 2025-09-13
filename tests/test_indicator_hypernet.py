import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "indicator_hypernet",
    Path(__file__).resolve().parents[1] / "models" / "indicator_hypernet.py",
)
indicator_hypernet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indicator_hypernet)  # type: ignore
IndicatorHyperNet = indicator_hypernet.IndicatorHyperNet

spec_ai = importlib.util.spec_from_file_location(
    "auto_indicators",
    Path(__file__).resolve().parents[1] / "features" / "auto_indicators.py",
)
auto_indicators = importlib.util.module_from_spec(spec_ai)
spec_ai.loader.exec_module(auto_indicators)  # type: ignore


def f1_score(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yp == 1 and yt == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yp == 0 and yt == 1)
    return 0.0 if (tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)


def test_deterministic_generation(tmp_path):
    data = {"price": [1.0, 2.0, 3.0, 4.0]}
    model = IndicatorHyperNet(in_dim=2, seed=0)
    path = tmp_path / "ind.yaml"
    _, d1 = auto_indicators.generate(data, model, [0.5], [1.0], registry_path=path)
    _, d2 = auto_indicators.generate(data, model, [0.5], [1.0], registry_path=path)
    assert d1 == d2


def test_hypernet_improves_validation(tmp_path):
    price = [1, 2, 3, 2, 3, 4]
    df = {"price": price}
    target = [0, 1, 1, 0, 1, 1]
    base_pred = [1] * len(target)
    base_f1 = f1_score(target, base_pred)

    model = IndicatorHyperNet(in_dim=2, seed=0)
    out, desc = auto_indicators.generate(df, model, [0.1], [0.0], registry_path=tmp_path / "disc.yaml")
    lag_col = out[f"price_lag{desc['lag']}"]
    preds = [1 if (lag is not None and p > lag) else 0 for p, lag in zip(out["price"], lag_col)]
    f1 = f1_score(target, preds)
    assert f1 >= base_f1
    auto_indicators.persist(desc, {"f1": f1}, registry_path=tmp_path / "disc.yaml")
    stored = json.loads((tmp_path / "disc.yaml").read_text())
    assert stored and "f1" in stored[-1]["metrics"]
