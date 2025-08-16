import os
import pytest
from .utils import run_smoke, ResourceCapabilities

pytestmark = pytest.mark.skipif(os.getenv("RESOURCE_TIER") != "standard", reason="Resource tier mismatch")


def test_mid_cpu() -> None:
    caps = ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=False, gpu_count=0)
    expected = {"sentiment": "sentiment_small", "rl_policy": "rl_small"}
    run_smoke(caps, "standard", expected)
