"""Utilities for introspecting the application configuration schema."""

from __future__ import annotations

from typing import Any, Iterator, Tuple, Type, get_args, get_origin
from typing import Annotated

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from mt5.config_models import AppConfig

__all__ = ["ConfigSchema", "iter_config_fields"]


class ConfigSchema(AppConfig):
    """Backward-compatible alias for :class:`config_models.AppConfig`."""


def _unwrap_base_model(annotation: Any) -> Type[BaseModel] | None:
    """Return the first ``BaseModel`` subtype contained in ``annotation``."""

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation

    origin = get_origin(annotation)
    if origin is None:
        return None

    if origin is Annotated:
        annotated_type = get_args(annotation)
        if annotated_type:
            return _unwrap_base_model(annotated_type[0])
        return None

    args = [
        arg
        for arg in get_args(annotation)
        if arg is not type(None)  # noqa: E721 - intentional identity comparison
    ]
    for arg in args:
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def iter_config_fields(
    model: Type[BaseModel] = ConfigSchema,
    prefix: str = "",
) -> Iterator[Tuple[str, FieldInfo]]:
    """Yield leaf ``FieldInfo`` objects from ``model`` and its nested models."""

    for name, field in model.model_fields.items():
        nested_model = _unwrap_base_model(field.annotation)
        if nested_model is not None:
            yield from iter_config_fields(nested_model, f"{prefix}{name}.")
            continue

        parameter = f"{prefix}{name}" if prefix else name
        yield parameter, field
