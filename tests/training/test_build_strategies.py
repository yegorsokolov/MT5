"""Tests for training.data.build_strategies."""

from training.data.build_strategies import generate_strategy_examples
from training.prompts.strategy_templates import epa_template


def test_generate_strategy_examples_contains_epa_sections():
    examples = generate_strategy_examples(epa_template, 3)
    assert len(examples) == 3
    for example in examples:
        assert example["evaluate"], "evaluate section missing"
        assert example["plan"], "plan section missing"
        assert example["act"], "act section missing"
