"""Model utilities and implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import joblib


class LazyModel:
    """Wrapper that defers weight loading until first use.

    Parameters
    ----------
    weight_path:
        Path to the serialized model weights. If ``None`` a ``loader`` callable
        must be supplied.
    loader:
        Callable used to construct the model. When ``weight_path`` is provided
        the callable receives the path as its sole argument. By default
        :func:`joblib.load` is used.
    """

    def __init__(
        self,
        weight_path: str | Path | None = None,
        loader: Callable[..., Any] | None = None,
    ) -> None:
        self.weight_path = Path(weight_path) if weight_path is not None else None
        self.loader: Callable[..., Any]
        if loader is None:
            self.loader = joblib.load
        else:
            self.loader = loader
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        """Return True if the underlying model has been materialised."""

        return self._model is not None

    def load(self) -> Any:
        """Load and return the underlying model."""

        if self._model is None:
            if self.weight_path is not None:
                self._model = self.loader(self.weight_path)
            else:
                self._model = self.loader()
        return self._model

    def unload(self) -> None:
        """Forget the loaded model allowing memory to be reclaimed."""

        self._model = None

    def set(self, model: Any) -> None:
        """Manually set the underlying model instance."""

        self._model = model

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple proxy
        model = self.load()
        return getattr(model, name)


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
try:  # optional torch dependency
    from .slimmable_network import SlimmableNetwork
except Exception:  # pragma: no cover - torch may be missing
    SlimmableNetwork = None
try:  # optional torch dependency
    from .cross_asset_transformer import CrossAssetTransformer
except Exception:  # pragma: no cover - torch may be missing
    CrossAssetTransformer = None

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
    "SlimmableNetwork",
    "CrossAssetTransformer",
    "LazyModel",
]
