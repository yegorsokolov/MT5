"""Unit tests for training.prompt.strategy_templates."""

from training.prompts.strategy_templates import epa_template


def test_epa_template_sections_populated():
    """Ensure the evaluate, plan and act sections are returned and non-empty."""
    result = epa_template("goal", "context")

    assert result["evaluate"], "evaluate section should not be empty"
    assert result["plan"], "plan section should not be empty"
    assert result["act"], "act section should not be empty"
