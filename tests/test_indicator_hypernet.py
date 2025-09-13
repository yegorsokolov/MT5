import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from models.indicator_hypernet import IndicatorHyperNet


def _base_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    true = np.linspace(0, 1, n)
    price = true + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"price": price, "market_regime": 0})
    df["target"] = np.append(true[1:], true[-1])
    return df


def test_hypernet_deterministic(tmp_path: Path) -> None:
    df = _base_df()
    hn1 = IndicatorHyperNet(tmp_path)
    _, gen1 = hn1.apply_or_generate(df.copy())
    hn2 = IndicatorHyperNet(tmp_path / "other")
    _, gen2 = hn2.apply_or_generate(df.copy())
    assert gen1 == gen2


def test_hypernet_validation_improves(tmp_path: Path) -> None:
    df = _base_df(100)
    hn = IndicatorHyperNet(tmp_path)
    df2, gen = hn.apply_or_generate(df.copy())
    name = next(iter(gen))
    baseline = ((df2["target"] - df2["price"]) ** 2).mean()
    indicator = ((df2["target"] - df2[name]) ** 2).mean()
    assert indicator < baseline
    assert hn.log_path.exists() is False  # no log yet
    hn.log_performance(name, float(baseline - indicator))
    data = json.loads(hn.log_path.read_text().splitlines()[-1])
    assert data["name"] == name
