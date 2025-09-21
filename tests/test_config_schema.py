import pytest
from pydantic import ValidationError

from config_models import AppConfig
from config_schema import iter_config_fields


def test_invalid_risk_per_trade():
    with pytest.raises(ValidationError) as exc:
        AppConfig(strategy={"risk_per_trade": -0.1, "symbols": ["XAUUSD"]})
    assert "risk_per_trade" in str(exc.value)


def test_symbols_must_be_list():
    with pytest.raises(ValidationError) as exc:
        AppConfig(strategy={"risk_per_trade": 0.1, "symbols": "XAUUSD"})
    assert "symbols" in str(exc.value)


def test_iter_config_fields_flattens_sections():
    fields = dict(iter_config_fields())
    assert "training.seed" in fields
    assert "strategy.risk_per_trade" in fields


def test_training_batch_fields_are_validated():
    cfg = AppConfig.model_validate(
        {
            "strategy": {"risk_per_trade": 0.1, "symbols": ["XAUUSD"]},
            "training": {
                "batch_size": 64,
                "eval_batch_size": 32,
                "min_batch_size": 16,
                "online_batch_size": 2048,
                "n_jobs": 4,
            },
        }
    )

    assert cfg.training.batch_size == 64
    assert cfg.training.eval_batch_size == 32
    assert cfg.training.min_batch_size == 16
    assert cfg.training.online_batch_size == 2048
    assert cfg.training.n_jobs == 4
    assert cfg.get("online_batch_size") == 2048


def test_training_batch_fields_reject_invalid_values():
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "strategy": {"risk_per_trade": 0.1, "symbols": ["XAUUSD"]},
                "training": {"batch_size": 0},
            }
        )
