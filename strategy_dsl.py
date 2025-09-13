from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import List, Sequence, Union, Iterable


@dataclass
class Buy:
    """Represents a buy action."""


@dataclass
class Sell:
    """Represents a sell action."""


@dataclass
class Wait:
    """Represents a no-op action."""


@dataclass
class Indicator:
    """Conditional primitive comparing two values from the market data."""

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


Token = Union[Buy, Sell, Wait, Indicator]


class StrategyInterpreter:
    """Executes a DSL program on sequential market data."""

    def __init__(self, program: Sequence[Token] | None = None):
        self.program = list(program) if program is not None else []

    def run(self, data: Iterable[dict], program: Sequence[Token] | None = None) -> float:
        program = list(program) if program is not None else self.program
        cash = 0.0
        position = 0
        data = list(data)
        if len(program) == len(data):
            for bar, token in zip(data, program):
                cash, position = self._execute(token, bar, cash, position)
        else:
            for bar in data:
                idx = 0
                while idx < len(program):
                    token = program[idx]
                    if isinstance(token, Indicator):
                        if idx + 1 >= len(program):
                            break
                        action = program[idx + 1]
                        if token.evaluate(bar):
                            cash, position = self._execute(action, bar, cash, position)
                        idx += 2
                    else:
                        cash, position = self._execute(token, bar, cash, position)
                        idx += 1
        if data:
            cash += position * data[-1]["price"]
        return cash

    @staticmethod
    def _execute(token: Token, bar: dict, cash: float, position: int) -> tuple[float, int]:
        price = bar["price"]
        if isinstance(token, Buy):
            if position == 0:
                position = 1
                cash -= price
        elif isinstance(token, Sell):
            if position == 1:
                position = 0
                cash += price
        # Wait does nothing
        return cash, position
