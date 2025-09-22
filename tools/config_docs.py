#!/usr/bin/env python3
"""Generate configuration documentation from the schema."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mt5.config_models import AppConfig
from mt5.config_schema import iter_config_fields
import html


def _range(field) -> str:
    parts: list[str] = []
    for meta in field.metadata:
        if hasattr(meta, "gt"):
            parts.append(f"> {meta.gt}")
        if hasattr(meta, "ge"):
            parts.append(f">= {meta.ge}")
        if hasattr(meta, "lt"):
            parts.append(f"< {meta.lt}")
        if hasattr(meta, "le"):
            parts.append(f"<= {meta.le}")
    return ", ".join(dict.fromkeys(parts))


def main() -> None:
    rows = []
    for name, field in iter_config_fields(AppConfig):
        default = (
            "required" if field.is_required() else field.get_default(call_default_factory=True)
        )
        rows.append(
            {
                "parameter": name,
                "description": field.description or "",
                "default": default,
                "range": _range(field),
            }
        )

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    md_lines = [
        "# Configuration Options",
        "",
        "| Parameter | Description | Default | Valid Range |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        rng = f"**{row['range']}**" if row["range"] else ""
        md_lines.append(
            f"| {row['parameter']} | {row['description']} | **{row['default']}** | {rng} |"
        )
    (docs_dir / "config.md").write_text("\n".join(md_lines) + "\n")

    html_lines = [
        "<table>",
        "<thead><tr><th>Parameter</th><th>Description</th><th>Default</th><th>Valid Range</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        rng = f"<strong>{html.escape(row['range'])}</strong>" if row["range"] else ""
        html_lines.append(
            "<tr><td>{}</td><td>{}</td><td><strong>{}</strong></td><td>{}</td></tr>".format(
                html.escape(row["parameter"]),
                html.escape(row["description"]),
                html.escape(str(row["default"])),
                rng,
            )
        )
    html_lines.append("</tbody></table>")
    (docs_dir / "config.html").write_text("\n".join(html_lines) + "\n")


if __name__ == "__main__":
    main()
