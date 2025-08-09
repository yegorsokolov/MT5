#!/usr/bin/env python3
"""Generate configuration documentation from the schema."""
from __future__ import annotations

from pathlib import Path
import sys

from annotated_types import Ge, Gt, Le, Lt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config_schema import ConfigSchema
import html


def _range(field) -> str:
    parts: list[str] = []
    for m in field.metadata:
        if isinstance(m, Gt):
            parts.append(f"> {m.gt}")
        elif isinstance(m, Ge):
            parts.append(f">= {m.ge}")
        elif isinstance(m, Lt):
            parts.append(f"< {m.lt}")
        elif isinstance(m, Le):
            parts.append(f"<= {m.le}")
    return ", ".join(parts)


def main() -> None:
    rows = []
    for name, field in ConfigSchema.model_fields.items():
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
