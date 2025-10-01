"""Lightweight YAML loader/dumper used in tests when PyYAML is unavailable.

The real project depends on :mod:`pyyaml`, but the exercise environment keeps
its dependency footprint intentionally small.  A number of tests import
``yaml.safe_load``/``yaml.safe_dump`` directly, so we provide a tiny,
feature-limited implementation that understands the subset of YAML these tests
use (nested mappings, lists of scalars and primitive values).

It is *not* a complete YAML parser – the goal is only to keep the lightweight
fixtures working without pulling in external wheels.  When PyYAML is installed
it will shadow this module.
"""
from __future__ import annotations

from io import TextIOBase
from typing import Tuple, Union, Any

Scalar = Union[str, int, float, bool, None]


class Mapping(dict):
    """Lightweight dictionary subclass for type checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Sequence(list):
    """Lightweight list subclass for type checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _parse_scalar(token: str) -> Scalar:
    token = token.strip()
    if not token:
        return ""
    lowered = token.lower()
    if lowered == "null" or lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    # numeric detection – try int before float so integers remain integers
    try:
        if token.startswith("0") and token not in {"0", "0.0"} and not token.startswith("0."):
            raise ValueError
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            pass
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    return token


def _preprocess(text: str) -> list[Tuple[int, str]]:
    lines: list[Tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in raw_line:
            hash_index = raw_line.find("#")
            if hash_index != -1 and raw_line[:hash_index].strip():
                stripped = raw_line[:hash_index].rstrip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, stripped))
    return lines


def _parse_sequence(lines: list[Tuple[int, str]], start: int, indent: int) -> Tuple[Sequence, int]:
    items: Sequence = Sequence()
    index = start
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError("Invalid indentation inside sequence")
        if not content.startswith("- ") and content != "-":
            break
        item_content = content[1:].lstrip()
        if not item_content:
            # Nested block (e.g. list of mappings) – parse as mapping
            child, next_index = _parse_mapping(lines, index + 1, indent + 2)
            items.append(child)
            index = next_index
            continue
        scalar = _parse_scalar(item_content)
        index += 1
        # Allow optional nested block extending the scalar into mapping
        if index < len(lines) and lines[index][0] > indent:
            child, next_index = _parse_mapping(lines, index, indent + 2)
            if isinstance(scalar, Mapping):
                scalar.update(child)
                items.append(scalar)
            elif isinstance(child, Mapping) and isinstance(scalar, str):
                # Interpret "key: value" encoded as "- key" followed by mapping
                child.setdefault("value", scalar)
                items.append(child)
            else:
                items.append(child)
            index = next_index
        else:
            items.append(scalar)
    return items, index


def _parse_mapping(lines: list[Tuple[int, str]], start: int, indent: int) -> Tuple[Mapping, int]:
    mapping: Mapping = Mapping()
    index = start
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError("Invalid indentation inside mapping")
        if content.startswith("- "):
            # This handles top-level sequences without explicit key
            seq, next_index = _parse_sequence(lines, index, indent)
            return Mapping({"items": seq}), next_index
        key, sep, rest = content.partition(":")
        key = key.strip()
        rest = rest.strip()
        index += 1
        if not sep:
            raise ValueError(f"Malformed line: {content}")
        if rest:
            mapping[key] = _parse_scalar(rest)
        else:
            if index < len(lines) and lines[index][0] > indent:
                next_line_indent, next_content = lines[index]
                if next_content.startswith("- ") or next_content == "-":
                    seq, index = _parse_sequence(lines, index, indent + 2)
                    mapping[key] = seq
                else:
                    child, index = _parse_mapping(lines, index, indent + 2)
                    mapping[key] = child
            else:
                mapping[key] = None
    return mapping, index


def safe_load(stream: Union[str, TextIOBase]) -> Any:
    """Parse a tiny subset of YAML into Python primitives."""

    if hasattr(stream, "read"):
        text = stream.read()
    else:
        text = str(stream)
    lines = _preprocess(text)
    if not lines:
        return {}
    mapping, index = _parse_mapping(lines, 0, lines[0][0])
    if index != len(lines):
        # Remaining lines belong to a top-level sequence or mapping at root indent 0
        if lines[index][0] == 0 and lines[index][1].startswith("- "):
            seq, _ = _parse_sequence(lines, index, 0)
            return seq
    return dict(mapping)


def _format_scalar(value: Scalar) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _as_mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else Mapping(value)


def _as_sequence(value: Any) -> Sequence:
    return value if isinstance(value, Sequence) else Sequence(value)


def _dump_mapping(data: Mapping, indent: int, lines: list[str]) -> None:
    for key, value in data.items():
        if isinstance(value, (Mapping, dict)):
            lines.append(" " * indent + f"{key}:")
            _dump_mapping(_as_mapping(value), indent + 2, lines)
        elif isinstance(value, (Sequence, list)):
            lines.append(" " * indent + f"{key}:")
            _dump_sequence(_as_sequence(value), indent + 2, lines)
        else:
            lines.append(" " * indent + f"{key}: {_format_scalar(value)}")


def _dump_sequence(data: Sequence, indent: int, lines: list[str]) -> None:
    for item in data:
        if isinstance(item, (Mapping, dict)):
            lines.append(" " * indent + "-")
            _dump_mapping(_as_mapping(item), indent + 2, lines)
        elif isinstance(item, (Sequence, list)):
            lines.append(" " * indent + "-")
            _dump_sequence(_as_sequence(item), indent + 2, lines)
        else:
            lines.append(" " * indent + f"- {_format_scalar(item)}")


def safe_dump(data: Any, stream: TextIOBase | None = None, **_: Any) -> str | None:
    """Serialize ``data`` to YAML using the supported subset."""

    if isinstance(data, dict) and not isinstance(data, Mapping):
        mapping = Mapping(data)
    elif isinstance(data, Mapping):
        mapping = data
    else:
        raise TypeError("safe_dump expects a mapping at the top level")

    lines: list[str] = []
    _dump_mapping(mapping, 0, lines)
    output = "\n".join(lines) + ("\n" if lines else "")
    if stream is not None:
        stream.write(output)
        return None
    return output


__all__ = ["safe_load", "safe_dump"]
