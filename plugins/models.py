from . import register_model


@register_model(name="dummy_model", tier="full")
def load_full_model():
    """Return identifier for the full model variant."""
    return "full_model"


@register_model(name="dummy_model", tier="lite")
def load_lite_model():
    """Return identifier for the lite model variant."""
    return "lite_model"

