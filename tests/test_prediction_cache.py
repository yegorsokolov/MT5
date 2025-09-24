import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


class _Counter:
    def __init__(self) -> None:
        self._value = types.SimpleNamespace(
            val=0, set=lambda v: setattr(self._value, "val", v), get=lambda: self._value.val
        )

    def inc(self) -> None:
        self._value.set(self._value.get() + 1)


class _Gauge:
    def __init__(self) -> None:
        self.value = 0.0

    def set(self, v: float) -> None:
        self.value = v


@contextmanager
def _patched_prediction_cache():
    from mt5 import metrics as metrics_module

    original_hit = metrics_module.PRED_CACHE_HIT
    original_ratio = metrics_module.PRED_CACHE_HIT_RATIO
    counter = _Counter()
    gauge = _Gauge()
    metrics_module.PRED_CACHE_HIT = counter
    metrics_module.PRED_CACHE_HIT_RATIO = gauge

    prediction_cache_module = importlib.import_module("mt5.prediction_cache")
    prediction_cache_module = importlib.reload(prediction_cache_module)
    try:
        yield prediction_cache_module.PredictionCache, counter, gauge
    finally:
        metrics_module.PRED_CACHE_HIT = original_hit
        metrics_module.PRED_CACHE_HIT_RATIO = original_ratio
        importlib.reload(prediction_cache_module)


def test_prediction_cache_lru_eviction():
    with _patched_prediction_cache() as (PredictionCache, hit_metric, ratio_metric):
        hit_metric._value.set(0)
        ratio_metric.value = 0.0
        cache = PredictionCache(maxsize=2, policy="lru")
        cache.set(1, 0.1)
        cache.set(2, 0.2)
        assert cache.get(1) == 0.1
        cache.set(3, 0.3)  # evicts key 2
        assert cache.get(2) is None
        assert hit_metric._value.get() == 1
        assert ratio_metric.value == 0.5


def test_prediction_cache_fifo_eviction():
    with _patched_prediction_cache() as (PredictionCache, hit_metric, ratio_metric):
        hit_metric._value.set(0)
        ratio_metric.value = 0.0
        cache = PredictionCache(maxsize=2, policy="fifo")
        cache.set(1, 0.1)
        cache.set(2, 0.2)
        assert cache.get(1) == 0.1
        cache.set(3, 0.3)  # evicts oldest (key 1)
        assert cache.get(1) is None
        # hit metric incremented from earlier get
        assert hit_metric._value.get() == 1
        assert ratio_metric.value == 0.5


def test_prediction_cache_without_metrics_module():
    metrics_module = sys.modules.get("mt5.metrics")
    prediction_cache_module = sys.modules.get("mt5.prediction_cache")

    try:
        sys.modules["mt5.metrics"] = None
        sys.modules.pop("mt5.prediction_cache", None)
        fallback_module = importlib.import_module("mt5.prediction_cache")

        cache = fallback_module.PredictionCache(maxsize=2, policy="lru")
        cache.set(1, "alpha")
        cache.set(2, "beta")
        assert cache.get(1) == "alpha"
        assert cache.get(2) == "beta"
    finally:
        if metrics_module is not None:
            sys.modules["mt5.metrics"] = metrics_module
        else:
            sys.modules.pop("mt5.metrics", None)

        if prediction_cache_module is not None:
            sys.modules["mt5.prediction_cache"] = prediction_cache_module
        else:
            sys.modules.pop("mt5.prediction_cache", None)

        restored_module = importlib.import_module("mt5.prediction_cache")
        importlib.reload(restored_module)

