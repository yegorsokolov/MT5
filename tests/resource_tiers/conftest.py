import pytest
from .utils import ResourceCapabilities


@pytest.fixture
def lite_caps() -> ResourceCapabilities:
    return ResourceCapabilities(cpus=2, memory_gb=4, has_gpu=False, gpu_count=0)


@pytest.fixture
def standard_caps() -> ResourceCapabilities:
    return ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=False, gpu_count=0)


@pytest.fixture
def gpu_caps() -> ResourceCapabilities:
    return ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
