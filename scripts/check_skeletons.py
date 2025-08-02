#!/usr/bin/env python3
"""Detect placeholder code left in the repository.

The script walks the repository and searches Python files for:
- `pass` statements that are not part of an ``except`` block,
- comments containing ``TODO``,
- occurrences of ``NotImplementedError``.

If any such occurrences are found outside of known exceptions the script
prints them and exits with status 1. It is intended to be used in CI and as a
local pre-flight check before releases.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable

# Paths (relative to repo root) that are allowed to contain placeholder code.
# ``ray/__init__.py`` is a small stub used for testing without the real Ray
# dependency and intentionally contains ``pass`` statements.
KNOWN_EXCEPTIONS = {
    Path("ray/__init__.py"),
    Path("scripts/check_skeletons.py"),
}

EXCLUDE_DIRS = {
    ".git",
    ".github",
    "proto",
    "__pycache__",
    "tests",
}

KEYWORDS = ("TODO", "NotImplementedError")


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if rel in KNOWN_EXCEPTIONS:
            continue
        yield path


class PassVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: list[ast.AST] = []
        self.linenos: list[int] = []

    def generic_visit(self, node: ast.AST) -> None:  # type: ignore[override]
        self.parents.append(node)
        super().generic_visit(node)
        self.parents.pop()

    def visit_Pass(self, node: ast.Pass) -> None:  # type: ignore[override]
        parent = self.parents[-1] if self.parents else None
        if not isinstance(parent, ast.ExceptHandler):
            self.linenos.append(node.lineno)


def check_file(path: Path) -> list[str]:
    results: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - unlikely
        results.append(f"{path}: unable to read file ({exc})")
        return results

    # Check for TODO and NotImplementedError occurrences
    for lineno, line in enumerate(text.splitlines(), start=1):
        if any(keyword in line for keyword in KEYWORDS):
            results.append(f"{path}:{lineno}: {line.strip()}")

    # Analyse AST for pass statements
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        results.append(f"{path}:{exc.lineno}: syntax error")
        return results

    visitor = PassVisitor()
    visitor.visit(tree)
    for lineno in visitor.linenos:
        results.append(f"{path}:{lineno}: pass statement")
    return results


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    findings: list[str] = []
    for file in iter_python_files(root):
        findings.extend(check_file(file))

    if findings:
        print("Skeleton code detected:")
        for item in findings:
            print("  " + item)
        return 1
    print("No skeleton code found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
