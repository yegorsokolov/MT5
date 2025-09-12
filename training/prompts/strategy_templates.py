"""Prompt strategy templates for agent training.

This module defines high-level prompt templates used to guide an
agent through a structured reasoning process. Currently only an
``epa_template`` is provided which separates reasoning into three
stages: *evaluate*, *plan* and *act*.
"""

from __future__ import annotations

from typing import Dict

from training.config import StrategyConfig


def epa_template(goal: str, context: str, config: StrategyConfig | None = None) -> Dict[str, str]:
    """Create an Evaluate-Plan-Act prompt template.

    The template divides reasoning into three sequential sections:

    - ``evaluate``: Understand the goal and analyse any relevant context.
    - ``plan``: Devise a concise step-by-step strategy based on the evaluation.
    - ``act``: Execute the plan and produce a final answer or decision.

    Parameters
    ----------
    goal:
        The objective the agent should accomplish.
    context:
        Additional information that may influence reasoning about the goal.

    Returns
    -------
    Dict[str, str]
        A mapping containing the ``evaluate``, ``plan`` and ``act``
        sections of the template. Each value is a short instructional
        string describing the intent of that stage.
    """

    # Evaluate: break down the goal and consider constraints and relevant details.
    evaluate = (
        f"Evaluate the goal: {goal}.\n"  # Start by restating the goal for clarity.
        f"Context: {context}.\n"  # Provide context to ground the evaluation.
        "List key factors, constraints, and unknowns."
    )

    if config and config.budget_limit is not None:
        evaluate += f" Budget limit: {config.budget_limit}."
    if config and config.risk_tolerance is not None:
        evaluate += f" Risk tolerance: {config.risk_tolerance}."

    # Plan: outline a strategy using insights from the evaluation stage.
    plan = (
        "Plan a step-by-step approach to achieve the goal. "
        "Address the factors identified during evaluation."
    )

    # Act: execute the plan and deliver the final outcome.
    act = (
        "Act on the plan and provide the resulting output or decision. "
        "Explain how the actions address the goal."
    )

    sections = {"evaluate": evaluate, "plan": plan, "act": act}
    if config and config.custom_sections:
        sections.update(config.custom_sections)
    return sections

