import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import update_config

@pytest.mark.parametrize('key', [
    'max_daily_loss',
    'max_drawdown',
    'max_var',
    'max_stress_loss',
    'max_cvar',
    'risk_per_trade',
    'rl_max_position',
])
def test_update_protected_key_raises(key):
    with pytest.raises(ValueError):
        update_config(key, 0, 'test')
