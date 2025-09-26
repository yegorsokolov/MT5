try:  # optional heavy dependency stack
    from .router import StrategyRouter
except Exception:  # pragma: no cover - allow lightweight environments
    StrategyRouter = None  # type: ignore

try:
    from .fuzzy_consensus import FuzzyConsensus
except Exception:  # pragma: no cover - optional dependency missing
    FuzzyConsensus = None  # type: ignore

from .archive import StrategyArchive

try:  # optional imports to avoid heavy dependencies during tests
    from .performance_monitor import PerformanceMonitor
    from .bayesian_weighting import BayesianWeighting
    from .evolution_lab import EvolutionLab
except Exception:  # pragma: no cover - optional at runtime
    PerformanceMonitor = BayesianWeighting = EvolutionLab = None

__all__ = [
    "StrategyRouter",
    "PerformanceMonitor",
    "BayesianWeighting",
    "EvolutionLab",
    "FuzzyConsensus",
    "StrategyArchive",
]
