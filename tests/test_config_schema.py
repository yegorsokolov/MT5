import pytest
from pydantic import ValidationError

from config_models import AppConfig


def test_invalid_risk_per_trade():
    with pytest.raises(ValidationError) as exc:
        AppConfig(strategy={"risk_per_trade": -0.1, "symbols": ["XAUUSD"]})
    assert "risk_per_trade" in str(exc.value)


def test_symbols_must_be_list():
    with pytest.raises(ValidationError) as exc:
        AppConfig(strategy={"risk_per_trade": 0.1, "symbols": "XAUUSD"})
    assert "symbols" in str(exc.value)
