import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategy.router import StrategyRouter


def test_router_uses_rationale_scores(tmp_path):
    win_rates = pd.DataFrame(
        {"algorithm": ["a", "b"], "win_rate": [0.9, 0.1]}
    ).set_index("algorithm")
    path = tmp_path / "win.parquet"
    win_rates.to_parquet(path)

    router = StrategyRouter(
        algorithms={"a": lambda f: 0.0, "b": lambda f: 0.0},
        alpha=0.0,
        rationale_path=path,
    )

    choice = router.select({"volatility": 0.0, "trend_strength": 0.0, "regime": 0.0})
    assert choice == "a"
