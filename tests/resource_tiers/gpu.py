import os
import pytest
from .utils import run_smoke, ResourceCapabilities

pytestmark = pytest.mark.skipif(os.getenv("RESOURCE_TIER") != "gpu", reason="Resource tier mismatch")


def test_gpu() -> None:
    caps = ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
    expected = {"sentiment": "sentiment_large", "rl_policy": "rl_medium"}
    run_smoke(caps, "gpu", expected)
