from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Iterable


@dataclass
class Indicator:
    """Compares two fields from a market data bar."""

    lhs: str
    op: str
    rhs: Union[str, float, int]

    OPS = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def evaluate(self, bar: dict) -> bool:
        lhs_val = bar[self.lhs]
        rhs_val = bar[self.rhs] if isinstance(self.rhs, str) else self.rhs
        return self.OPS[self.op](lhs_val, rhs_val)


@dataclass
class Filter:
    """Pass-through filter that can drop a signal."""

    def apply(self, signal: bool) -> bool:
        return bool(signal)


@dataclass
class PositionSizer:
    """Determines position size when a signal is active."""

    size: float = 1.0


@dataclass
class ExitRule:
    """Exits any open position when triggered."""

    def should_exit(self, signal: bool) -> bool:
        return not signal


Node = Union[Indicator, Filter, PositionSizer, ExitRule]


@dataclass
class StrategyGraph:
    """Simple directed acyclic graph representing a trading strategy."""

    nodes: Dict[int, Node]
    edges: List[Tuple[int, int, Optional[bool]]]
    entry: int = 0

    def to_dict(self) -> dict:
        return {
            "nodes": {k: type(v).__name__ for k, v in self.nodes.items()},
            "edges": self.edges,
            "entry": self.entry,
        }

    @classmethod
    def from_dict(cls, data: dict, node_objs: Dict[int, Node]) -> "StrategyGraph":
        return cls(nodes=node_objs, edges=data["edges"], entry=data.get("entry", 0))

    def _next(self, current: int, result: Optional[bool]) -> Optional[int]:
        for src, dst, cond in self.edges:
            if src == current and (cond is None or cond == result):
                return dst
        return None

    def run(self, data: Iterable[dict]) -> float:
        cash = 0.0
        position = 0.0
        data = list(data)
        for bar in data:
            node_id: Optional[int] = self.entry
            result: Optional[bool] = None
            while node_id is not None:
                node = self.nodes[node_id]
                if isinstance(node, Indicator):
                    result = node.evaluate(bar)
                elif isinstance(node, Filter):
                    result = node.apply(result)
                elif isinstance(node, PositionSizer):
                    if result and position == 0.0:
                        position = node.size
                        cash -= bar["price"] * position
                elif isinstance(node, ExitRule):
                    if node.should_exit(result) and position > 0.0:
                        cash += bar["price"] * position
                        position = 0.0
                node_id = self._next(node_id, result)
        if data and position > 0.0:
            cash += data[-1]["price"] * position
        return float(cash)
