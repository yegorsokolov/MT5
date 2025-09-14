import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

proj_root = Path(__file__).resolve().parents[1]
sys.path.append(str(proj_root))

from feature_store import purge_version, register_feature, request_indicator
from strategies.graph_dsl import IndicatorNode, StrategyGraph

strategy_pkg = types.ModuleType("strategy")
strategy_pkg.__path__ = [str(proj_root / "strategy")]
sys.modules.setdefault("strategy", strategy_pkg)

spec = importlib.util.spec_from_file_location(
    "strategy.evolution_lab", proj_root / "strategy" / "evolution_lab.py"
)
evolution_lab = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["strategy.evolution_lab"] = evolution_lab
spec.loader.exec_module(evolution_lab)  # type: ignore
EvolutionLab = evolution_lab.EvolutionLab


def test_graph_mutation_and_indicator(tmp_path):
    df = pd.DataFrame({"price": [1, 2], "ma": [1, 1]})
    register_feature("mut_v1", df)
    cols = request_indicator("mut_v1")
    assert "price" in cols
    graph = StrategyGraph(
        nodes={0: IndicatorNode("price", ">", "ma")}, edges=[], entry=0
    )
    node = IndicatorNode("price", "<", "ma")
    nid = graph.insert_node(0, None, node)
    assert nid in graph.nodes
    graph.remove_node(nid)
    assert nid not in graph.nodes
    purge_version("mut_v1")


def test_evolve_queue_persist(tmp_path):
    lab = EvolutionLab(base=lambda msg: 0.1, lineage_path=tmp_path / "lineage.parquet")
    seeds = [0, 1, 2]
    msgs = [{"price": 1.0}]
    results = asyncio.run(lab.evolve_queue(seeds, msgs, concurrency=2))
    assert len(results) == 3
    path = tmp_path / "lineage.parquet"
    assert path.exists()
    assert path.stat().st_size > 0
