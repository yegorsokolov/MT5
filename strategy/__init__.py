from .router import StrategyRouter

try:  # optional imports to avoid heavy dependencies during tests
    from .performance_monitor import PerformanceMonitor
    from .bayesian_weighting import BayesianWeighting
    from .evolution_lab import EvolutionLab
except Exception:  # pragma: no cover - optional at runtime
    PerformanceMonitor = BayesianWeighting = EvolutionLab = None

__all__ = ["StrategyRouter", "PerformanceMonitor", "BayesianWeighting", "EvolutionLab"]
