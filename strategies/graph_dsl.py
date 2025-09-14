"""Simple DAG-style strategy description language.

This module defines a minimal directed acyclic graph (DAG) based DSL used for
describing trading strategies.  Strategies consist of **indicator** nodes that
produce boolean signals, optional **risk** nodes which can adjust position
size, and **position** nodes that open/close positions based on the signal.  A
small :class:`StrategyGraph` container executes the graph over market data.

Each node type supports ``to_dict``/``from_dict`` helpers allowing strategies to
be serialised and stored or transmitted easily.

The implementation purposefully remains lightweight so the unit tests can run
quickly while exercising the high level behaviour of the trading engine.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union


# ---------------------------------------------------------------------------
# Node implementations


@dataclass
class IndicatorNode:
    """Compare two fields from a market data bar."""

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

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "type": "IndicatorNode",
            "lhs": self.lhs,
            "op": self.op,
            "rhs": self.rhs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IndicatorNode:
        return cls(lhs=data["lhs"], op=data["op"], rhs=data["rhs"])


@dataclass
class RiskNode:
    """Adjust position size by a constant scale factor."""

    scale: float = 1.0

    def apply(self, signal: bool) -> bool:
        """Pass-through hook for backwards compatibility with ``Filter``."""

        return bool(signal)

    def adjust(self, size: float) -> float:
        return size * self.scale

    def to_dict(self) -> dict:
        return {"type": "RiskNode", "scale": self.scale}

    @classmethod
    def from_dict(cls, data: dict) -> RiskNode:
        return cls(scale=data.get("scale", 1.0))


@dataclass
class PositionNode:
    """Open or close positions based on the current signal."""

    size: float = 1.0

    def trade(
        self,
        signal: bool,
        bar: dict,
        position: float,
        cash: float,
        scale: float,
    ) -> Tuple[float, float]:
        final_size = self.size * scale
        if signal and position == 0.0:
            position = final_size
            cash -= bar["price"] * position
        elif not signal and position > 0.0:
            cash += bar["price"] * position
            position = 0.0
        return cash, position

    def to_dict(self) -> dict:
        return {"type": "PositionNode", "size": self.size}

    @classmethod
    def from_dict(cls, data: dict) -> PositionNode:
        return cls(size=data.get("size", 1.0))


@dataclass
class ExitNode:
    """Explicit exit node which closes any open position."""

    def should_exit(self, signal: bool) -> bool:
        return not signal

    def to_dict(self) -> dict:  # pragma: no cover - trivial
        return {"type": "ExitNode"}

    @classmethod
    def from_dict(cls, data: dict) -> ExitNode:  # pragma: no cover - trivial
        return cls()


Node = Union[IndicatorNode, RiskNode, PositionNode, ExitNode]


# Registry for de/serialisation
_NODE_TYPES: Dict[str, Type[Node]] = {
    "IndicatorNode": IndicatorNode,
    "RiskNode": RiskNode,
    "PositionNode": PositionNode,
    "ExitNode": ExitNode,
}


def node_from_dict(data: dict) -> Node:
    cls = _NODE_TYPES[data["type"]]
    return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Graph container


@dataclass
class StrategyGraph:
    """Directed acyclic graph representing a trading strategy."""

    nodes: Dict[int, Node]
    edges: List[Tuple[int, int, Optional[bool]]]
    entry: int = 0

    def to_dict(self) -> dict:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": self.edges,
            "entry": self.entry,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StrategyGraph:
        nodes = {int(k): node_from_dict(v) for k, v in data["nodes"].items()}
        edges = [tuple(edge) for edge in data["edges"]]
        return cls(nodes=nodes, edges=edges, entry=data.get("entry", 0))

    # ---- mutation ------------------------------------------------------
    def insert_node(
        self,
        src: int,
        dst: Optional[int],
        node: Node,
        cond: Optional[bool] = None,
    ) -> int:
        """Insert ``node`` between ``src`` and ``dst``.

        ``dst`` may be ``None`` in which case the new node becomes a leaf.  The
        method returns the new node identifier.
        """

        new_id = max(self.nodes, default=-1) + 1
        self.nodes[new_id] = node
        if dst is not None:
            self.edges = [
                e
                for e in self.edges
                if not (e[0] == src and e[1] == dst and (cond is None or e[2] == cond))
            ]
            self.edges.append((src, new_id, cond))
            self.edges.append((new_id, dst, None))
        else:
            self.edges.append((src, new_id, cond))
        return new_id

    def remove_node(self, node_id: int) -> None:
        """Remove ``node_id`` and any connected edges."""

        if node_id not in self.nodes:
            return
        self.nodes.pop(node_id)
        self.edges = [e for e in self.edges if e[0] != node_id and e[1] != node_id]
        if self.entry == node_id:
            self.entry = min(self.nodes.keys(), default=0)

    # ---- execution -----------------------------------------------------
    def _next(self, current: int, result: Optional[bool]) -> Optional[int]:
        for src, dst, cond in self.edges:
            if src == current and (cond is None or cond == result):
                return dst
        return None

    def run(self, data: Iterable[dict]) -> float:
        cash = 0.0
        position = 0.0
        data_list = list(data)
        for bar in data_list:
            node_id: Optional[int] = self.entry
            result: Optional[bool] = None
            scale = 1.0
            while node_id is not None:
                node = self.nodes[node_id]
                if isinstance(node, IndicatorNode):
                    result = node.evaluate(bar)
                elif isinstance(node, RiskNode):
                    scale = node.adjust(scale)
                    if result is not None:
                        result = node.apply(result)
                elif isinstance(node, PositionNode):
                    if result is not None:
                        cash, position = node.trade(result, bar, position, cash, scale)
                elif isinstance(node, ExitNode):
                    if node.should_exit(result) and position > 0.0:
                        cash += bar["price"] * position
                        position = 0.0
                node_id = self._next(node_id, result)
        if data_list and position > 0.0:
            cash += data_list[-1]["price"] * position
        return float(cash)


# ---------------------------------------------------------------------------
# Backwards compatibility aliases


Indicator = IndicatorNode
Filter = RiskNode
PositionSizer = PositionNode
ExitRule = ExitNode


__all__ = [
    "IndicatorNode",
    "RiskNode",
    "PositionNode",
    "ExitNode",
    "StrategyGraph",
    "node_from_dict",
    # aliases
    "Indicator",
    "Filter",
    "PositionSizer",
    "ExitRule",
]
