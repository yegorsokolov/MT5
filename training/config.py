from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class StrategyConfig:
    """Configuration options for strategy prompt generation.

    The dataclass is intentionally lightweight so callers can construct it
    from YAML or JSON configuration files and pass the resulting values to
    prompt builders.

    Parameters
    ----------
    budget_limit:
        Optional budget limit to include in prompts.
    risk_tolerance:
        Optional risk tolerance to include in prompts.
    custom_sections:
        Mapping of custom section names to instructional text.
    """

    budget_limit: Optional[float] = None
    risk_tolerance: Optional[float] = None
    custom_sections: Dict[str, str] = field(default_factory=dict)
