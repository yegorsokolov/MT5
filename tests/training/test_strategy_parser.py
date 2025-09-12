"""Tests for training.parsers.strategy_parser."""

from training.parsers.strategy_parser import StrategyParser


def test_parse_preserves_custom_tactics():
    parser = StrategyParser()
    data = {
        "evaluate": "Assess the situation",
        "plan": "Outline steps",
        "custom_tactics": "Leverage social engineering",
        "act": "Execute plan",
    }
    result = parser.parse(data)
    assert result.custom_tactics == "Leverage social engineering"
    assert result.evaluate == "Assess the situation"
    assert result.plan == "Outline steps"
    assert result.act == "Execute plan"
