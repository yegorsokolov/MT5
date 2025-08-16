from prediction_cache import PredictionCache
from metrics import PRED_CACHE_HIT


def test_prediction_cache_lru_eviction():
    PRED_CACHE_HIT._value.set(0)  # reset metric
    cache = PredictionCache(maxsize=2, policy="lru")
    cache.set(1, 0.1)
    cache.set(2, 0.2)
    assert cache.get(1) == 0.1
    cache.set(3, 0.3)  # evicts key 2
    assert cache.get(2) is None
    assert PRED_CACHE_HIT._value.get() == 1


def test_prediction_cache_fifo_eviction():
    PRED_CACHE_HIT._value.set(0)
    cache = PredictionCache(maxsize=2, policy="fifo")
    cache.set(1, 0.1)
    cache.set(2, 0.2)
    assert cache.get(1) == 0.1
    cache.set(3, 0.3)  # evicts oldest (key 1)
    assert cache.get(1) is None
    # hit metric incremented from earlier get
    assert PRED_CACHE_HIT._value.get() == 1

