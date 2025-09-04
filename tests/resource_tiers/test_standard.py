import os
import pytest
from .utils import run_smoke

pytestmark = pytest.mark.skipif(os.getenv("RESOURCE_TIER") != "standard", reason="Resource tier mismatch")


def test_mid_cpu(standard_caps) -> None:
    expected = {"sentiment": "sentiment_small", "rl_policy": "rl_small"}
    run_smoke(standard_caps, "standard", expected)
