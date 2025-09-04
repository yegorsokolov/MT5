import os
import pytest
from .utils import run_smoke

pytestmark = pytest.mark.skipif(os.getenv("RESOURCE_TIER") != "lite", reason="Resource tier mismatch")


def test_low_cpu(lite_caps) -> None:
    expected = {"sentiment": "sentiment_small_quantized", "rl_policy": "rl_small_quantized"}
    run_smoke(lite_caps, "lite", expected)
