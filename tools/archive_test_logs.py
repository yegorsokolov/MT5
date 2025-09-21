#!/usr/bin/env python3
"""Utility to bundle pytest log archives for CI or local inspection."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
import tarfile
import zipfile

LATEST_MARKER = "LATEST_RUN"


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def _current_commit(repo_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root)
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return output.decode().strip() or "unknown"


def _read_latest(commit_dir: Path) -> str | None:
    marker = commit_dir / LATEST_MARKER
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip() or None
    return None


def _list_runs(commit_dir: Path) -> list[Path]:
    runs: list[Path] = []
    if not commit_dir.exists():
        return runs
    for path in commit_dir.iterdir():
        if path.is_dir() and path.name != LATEST_MARKER:
            runs.append(path)
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _derive_output_path(
    repo_root: Path,
    commit: str,
    identifier: str,
    archive_format: str,
    scope: str,
    explicit_output: str | None,
) -> Path:
    if explicit_output is not None:
        output_path = Path(explicit_output)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        return output_path

    suffix = ".tar.gz" if archive_format == "gztar" else ".zip"
    sanitized_commit = commit.replace("/", "_") or "unknown"
    sanitized_identifier = identifier.replace("/", "_") or "run"
    if scope == "commit":
        name = f"{sanitized_commit}-logs"
    else:
        name = f"{sanitized_commit}-{sanitized_identifier}"
    return repo_root / "logs" / "test_runs" / f"{name}{suffix}"


def _ensure_not_within(target: Path, output_path: Path) -> None:
    try:
        output_path.resolve().relative_to(target.resolve())
    except ValueError:
        return
    raise ValueError("Output archive cannot be created inside the source directory")


def _write_github_output(archive_path: Path, runs: list[Path], scope: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    lines = [f"archive-path={archive_path}"]
    if scope == "run" and runs:
        lines.append(f"latest-run={runs[0].name}")
    elif scope == "commit":
        run_names = ",".join(p.name for p in runs)
        lines.append(f"all-runs={run_names}")
    with open(github_output, "a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _make_archive(target: Path, output_path: Path, archive_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_format == "gztar":
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(target, arcname=target.name)
    elif archive_format == "zip":
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            root_prefix = f"{target.name}/"
            zipf.writestr(root_prefix, "")
            for item in sorted(target.rglob("*")):
                rel = item.relative_to(target)
                arcname = Path(target.name) / rel
                if item.is_file():
                    zipf.write(item, arcname=str(arcname))
                elif item.is_dir():
                    iterator = item.iterdir()
                    try:
                        next(iterator)
                    except StopIteration:
                        zipf.writestr(f"{arcname.as_posix()}/", "")
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"Unsupported archive format: {archive_format}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=["run", "commit"],
        default="run",
        help="Archive the latest run (default) or the entire commit folder.",
    )
    parser.add_argument(
        "--run-id",
        help="Specific run identifier to archive when scope=run. Defaults to the latest run.",
    )
    parser.add_argument(
        "--commit",
        help="Override the commit hash used for locating logs/test_runs/<commit>.",
    )
    parser.add_argument(
        "--output",
        help="Explicit output path for the created archive.",
    )
    parser.add_argument(
        "--format",
        choices=["gztar", "zip"],
        default="gztar",
        help="Archive format to use (default: gztar).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions that would be taken without creating the archive.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = _find_repo_root(Path(__file__).resolve())
    commit = args.commit or _current_commit(repo_root)
    commit_dir = repo_root / "logs" / "test_runs" / commit

    runs = _list_runs(commit_dir)
    if args.scope == "run":
        run_id = args.run_id or _read_latest(commit_dir)
        if run_id is None and runs:
            run_id = runs[0].name
        if run_id is None:
            print("No archived test runs found to package.")
            return 0
        target = commit_dir / run_id
        if not target.exists():
            print(f"Requested run '{run_id}' does not exist under {commit_dir}.")
            return 0
        identifier = run_id
    else:
        if not commit_dir.exists():
            print(f"No archived logs found under {commit_dir}.")
            return 0
        target = commit_dir
        identifier = commit

    output_path = _derive_output_path(
        repo_root=repo_root,
        commit=commit,
        identifier=identifier,
        archive_format=args.format,
        scope=args.scope,
        explicit_output=args.output,
    )

    _ensure_not_within(target, output_path)

    if args.dry_run:
        print(f"[DRY RUN] Would archive {target} to {output_path} using format {args.format}.")
        return 0

    _make_archive(target, output_path, args.format)
    print(f"Created archive at {output_path}")
    _write_github_output(output_path, runs, args.scope)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution guard
    sys.exit(main())
