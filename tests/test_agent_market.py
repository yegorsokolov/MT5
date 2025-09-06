import pathlib
import sys

import pandas as pd

# Ensure repository root on path for direct invocation
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from simulation.agent_market import AgentMarketSimulator


def test_agent_market_reproducibility(tmp_path):
    sim1 = AgentMarketSimulator(seed=123, steps=100, out_dir=tmp_path)
    trades1, book1 = sim1.run()
    sim2 = AgentMarketSimulator(seed=123, steps=100, out_dir=tmp_path)
    trades2, book2 = sim2.run()
    pd.testing.assert_frame_equal(trades1, trades2)
    pd.testing.assert_frame_equal(book1, book2)
    # ensure files are written
    assert (tmp_path / "trades_seed123.csv").exists()
    assert (tmp_path / "prices_seed123.csv").exists()
