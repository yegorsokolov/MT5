import os
import pytest
from .utils import run_smoke

pytestmark = pytest.mark.skipif(os.getenv("RESOURCE_TIER") != "gpu", reason="Resource tier mismatch")


def test_gpu(gpu_caps) -> None:
    expected = {"sentiment": "sentiment_large", "rl_policy": "rl_medium"}
    run_smoke(gpu_caps, "gpu", expected)
