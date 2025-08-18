"""Model utilities and implementations."""

try:  # optional heavy dependencies
    from .ensemble import EnsembleModel
except Exception:  # pragma: no cover
    EnsembleModel = None
try:  # pragma: no cover
    from .export import export_lightgbm, export_pytorch
except Exception:  # pragma: no cover
    export_lightgbm = export_pytorch = None
try:  # optional dependency
    from .price_distribution import PriceDistributionModel
except Exception:  # pragma: no cover - torch may be missing
    PriceDistributionModel = None
try:  # optional torch dependency
    from .meta_learner import MetaLearner
except Exception:  # pragma: no cover - torch may be missing
    MetaLearner = None
try:  # optional torch dependency
    from .multi_head import MultiHeadTransformer
except Exception:  # pragma: no cover - torch may be missing
    MultiHeadTransformer = None
try:  # optional torch dependency
    from .hier_forecast import HierarchicalForecaster
except Exception:  # pragma: no cover - torch may be missing
    HierarchicalForecaster = None
try:  # optional torch dependency
    from .tft import TemporalFusionTransformer, TFTConfig, QuantileLoss
except Exception:  # pragma: no cover - torch may be missing
    TemporalFusionTransformer = TFTConfig = QuantileLoss = None
from . import conformal

__all__ = [
    "EnsembleModel",
    "export_lightgbm",
    "export_pytorch",
    "MetaLearner",
    "PriceDistributionModel",
    "MultiHeadTransformer",
    "HierarchicalForecaster",
    "TemporalFusionTransformer",
    "TFTConfig",
    "QuantileLoss",
    "conformal",
]
