import os
import pytest
from .utils import run_smoke, ResourceCapabilities

pytestmark = pytest.mark.skipif(os.getenv("RESOURCE_TIER") != "lite", reason="Resource tier mismatch")


def test_low_cpu() -> None:
    caps = ResourceCapabilities(cpus=2, memory_gb=4, has_gpu=False, gpu_count=0)
    expected = {"sentiment": "sentiment_small", "rl_policy": "rl_small"}
    run_smoke(caps, "lite", expected)
