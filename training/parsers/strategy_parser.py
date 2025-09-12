"""Utilities for parsing strategy prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ParsedStrategy:
    """Dataclass representing a parsed strategy prompt.

    Attributes
    ----------
    evaluate:
        Text from the evaluation section.
    plan:
        Text from the planning section.
    act:
        Text from the action section.
    custom_tactics:
        Optional text describing any custom tactics or frameworks.
    """

    evaluate: str
    plan: str
    act: str
    custom_tactics: Optional[str] = None


class StrategyParser:
    """Parse strategy prompts produced by :func:`epa_template`.

    The parser expects a mapping with ``evaluate``, ``plan`` and ``act``
    keys and will optionally preserve a ``custom_tactics`` entry if
    present.
    """

    REQUIRED_KEYS = ("evaluate", "plan", "act")

    def parse(self, data: Dict[str, str]) -> ParsedStrategy:
        """Parse a mapping into a :class:`ParsedStrategy` instance.

        Parameters
        ----------
        data:
            Mapping containing strategy sections. Must include at least the
            ``evaluate``, ``plan`` and ``act`` keys.

        Returns
        -------
        ParsedStrategy
            Structured representation of the provided strategy.

        Raises
        ------
        ValueError
            If any required key is missing from ``data``.
        """

        missing = [key for key in self.REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"Missing required keys: {', '.join(missing)}")

        return ParsedStrategy(
            evaluate=data["evaluate"],
            plan=data["plan"],
            act=data["act"],
            custom_tactics=data.get("custom_tactics"),
        )
