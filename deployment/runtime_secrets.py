"""Generate runtime secrets for local services and tooling.

The trading bot exposes a lightweight configuration service and in-process
controller helpers that historically required manual secret management.  This
script generates the API keys and encryption material they expect, persisting
values to an environment file that systemd or container runtimes can source.
It mirrors the behaviour of :mod:`deployment.influx_bootstrap` so operators can
fully automate provisioning.

Example usage::

    python -m deployment.runtime_secrets \\
        --env-file deploy/secrets/runtime.env \\
        --print-exports

By default the helper is idempotent: existing secrets are reused unless
explicitly rotated via ``--rotate`` or ``--force``.  Additional sections can be
skipped with the ``--skip-*`` flags when an environment manages specific
secrets elsewhere (for instance via Vault).
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import secrets
import sys
import textwrap
from typing import Callable, Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class SecretSpec:
    """Specification for an individual secret that can be generated."""

    name: str
    generator: Callable[[], str]
    help: str


@dataclass(frozen=True)
class SectionSpec:
    """Group of related secrets that can be managed together."""

    slug: str
    title: str
    secrets: Sequence[SecretSpec]


DEFAULT_ENV_FILE = Path("deploy/secrets/runtime.env")


def _random_hex(num_bytes: int = 32) -> str:
    """Return a hexadecimal token with ``num_bytes`` of entropy."""

    return secrets.token_hex(num_bytes)


def _random_base64(num_bytes: int = 32) -> str:
    """Return a base64 encoded token suitable for AES or HMAC secrets."""

    return base64.b64encode(secrets.token_bytes(num_bytes)).decode("ascii")


SECTIONS: Sequence[SectionSpec] = (
    SectionSpec(
        slug="config",
        title="Configuration service API keys",
        secrets=(
            SecretSpec(
                name="CONFIG_ADMIN_KEY",
                generator=_random_hex,
                help="Admin credential granting read/write access to the configuration service.",
            ),
            SecretSpec(
                name="CONFIG_READER_KEY",
                generator=_random_hex,
                help="Read-only API key for the configuration service, used by dashboards and diagnostics.",
            ),
        ),
    ),
    SectionSpec(
        slug="controller",
        title="Local bot controller shared secrets",
        secrets=(
            SecretSpec(
                name="API_KEY",
                generator=_random_hex,
                help="x-api-key header clients must supply when invoking control-plane helpers.",
            ),
            SecretSpec(
                name="AUDIT_LOG_SECRET",
                generator=_random_base64,
                help="HMAC secret used to sign audit log entries produced by the controller.",
            ),
        ),
    ),
    SectionSpec(
        slug="encryption",
        title="AES-256 keys for encrypted artifacts",
        secrets=(
            SecretSpec(
                name="CHECKPOINT_AES_KEY",
                generator=_random_base64,
                help="Base64 encoded 32-byte key encrypting checkpoints under checkpoints/*.pkl.enc.",
            ),
            SecretSpec(
                name="DECISION_AES_KEY",
                generator=_random_base64,
                help="Base64 encoded 32-byte key protecting logs/decisions.parquet.enc.",
            ),
        ),
    ),
)

SECTION_MAP: Dict[str, SectionSpec] = {section.slug: section for section in SECTIONS}
ALL_SECRET_NAMES: frozenset[str] = frozenset(
    spec.name for section in SECTIONS for spec in section.secrets
)


@dataclass
class RuntimeSecretsResult:
    """Outcome of ensuring runtime secrets exist on disk."""

    path: Path
    values: Dict[str, str]
    created: List[str]
    rotated: List[str]
    written: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate runtime secrets for MT5 services")
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to the environment file that should receive the generated secrets (default: deploy/secrets/runtime.env)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rotate all managed secrets even when values already exist.",
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Do not manage configuration service API keys (CONFIG_ADMIN_KEY/CONFIG_READER_KEY).",
    )
    parser.add_argument(
        "--skip-controller",
        action="store_true",
        help="Do not manage local controller secrets (API_KEY/AUDIT_LOG_SECRET).",
    )
    parser.add_argument(
        "--skip-encryption",
        action="store_true",
        help="Do not manage AES encryption keys (CHECKPOINT_AES_KEY/DECISION_AES_KEY).",
    )
    parser.add_argument(
        "--rotate",
        action="append",
        default=[],
        metavar="NAME",
        help="Rotate a specific secret (may be specified multiple times).",
    )
    parser.add_argument(
        "--print-exports",
        action="store_true",
        help="Print the managed secrets as 'export KEY=VALUE' for shell usage.",
    )
    return parser.parse_args(argv)


def _read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _wrap_comment(text: str) -> List[str]:
    return [f"# {line}" for line in textwrap.wrap(text, width=88)] or [f"# {text}"]


def _write_env_file(
    path: Path,
    *,
    sections: Sequence[SectionSpec],
    managed_values: Mapping[str, str],
    existing: Mapping[str, str],
    force: bool,
) -> bool:
    """Persist ``managed_values`` to ``path`` while preserving existing entries."""

    merged: Dict[str, str] = dict(existing)
    changed = False
    for key, value in managed_values.items():
        current = merged.get(key)
        if current == value:
            continue
        if current is not None and not force:
            raise RuntimeError(
                f"Secret {key} already exists – rerun with --rotate {key} or --force to overwrite."
            )
        merged[key] = value
        changed = True
    if not changed and all(key in merged for key in managed_values):
        return False

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    managed_keys = {spec.name for section in sections for spec in section.secrets}
    lines: List[str] = [
        "# Runtime secrets generated by deployment.runtime_secrets",
        f"# Updated {timestamp}",
        "#",
    ]
    for section in sections:
        lines.append(f"# {section.title}")
        for spec in section.secrets:
            if spec.name not in merged:
                continue
            lines.extend(_wrap_comment(spec.help))
            lines.append(f"{spec.name}={merged[spec.name]}")
            lines.append("")
    preserved = [
        (key, merged[key])
        for key in sorted(merged)
        if key not in managed_keys and not key.startswith("#")
    ]
    if preserved:
        lines.append("# Additional entries preserved from previous runs")
        for key, value in preserved:
            lines.append(f"{key}={value}")
        lines.append("")
    content = "\n".join(lines).rstrip() + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return True


def ensure_runtime_secrets(
    env_path: Path,
    *,
    include_config: bool = True,
    include_controller: bool = True,
    include_encryption: bool = True,
    rotate: Iterable[str] | None = None,
    force: bool = False,
) -> RuntimeSecretsResult:
    """Ensure the requested runtime secrets exist in ``env_path``."""

    rotate_set = {item.strip().upper() for item in rotate or [] if item}
    unknown = rotate_set.difference(ALL_SECRET_NAMES)
    if unknown:
        raise RuntimeError(f"Unknown secrets requested for rotation: {', '.join(sorted(unknown))}")

    selected: List[SectionSpec] = []
    if include_config:
        selected.append(SECTION_MAP["config"])
    if include_controller:
        selected.append(SECTION_MAP["controller"])
    if include_encryption:
        selected.append(SECTION_MAP["encryption"])

    if not selected:
        return RuntimeSecretsResult(path=env_path, values=_read_env_file(env_path), created=[], rotated=[], written=False)

    existing = _read_env_file(env_path)
    managed: Dict[str, str] = {}
    created: List[str] = []
    rotated: List[str] = []

    for section in selected:
        for spec in section.secrets:
            current = existing.get(spec.name)
            should_rotate = force or spec.name in rotate_set or current is None
            value = spec.generator() if should_rotate else current
            if value is None:
                continue
            managed[spec.name] = value
            if current is None:
                created.append(spec.name)
            elif value != current:
                rotated.append(spec.name)

    unauthorised = [key for key in rotated if key not in rotate_set and not force]
    if unauthorised:
        raise RuntimeError(
            "Refusing to overwrite existing secrets: "
            + ", ".join(sorted(unauthorised))
            + ". Use --rotate NAME or --force to regenerate them."
        )

    final_values = dict(existing)
    final_values.update(managed)
    written = _write_env_file(
        env_path,
        sections=selected,
        managed_values=managed,
        existing=existing,
        force=force or bool(rotated),
    )
    if written and not final_values:
        final_values = _read_env_file(env_path)
    return RuntimeSecretsResult(
        path=env_path,
        values=final_values,
        created=created,
        rotated=rotated,
        written=written,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env_path = Path(args.env_file).expanduser()
    try:
        result = ensure_runtime_secrets(
            env_path,
            include_config=not args.skip_config,
            include_controller=not args.skip_controller,
            include_encryption=not args.skip_encryption,
            rotate=args.rotate,
            force=args.force,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.print_exports:
        for key in sorted(k for k in result.values if k in ALL_SECRET_NAMES):
            value = result.values[key]
            print(f"export {key}='{value}'")

    if result.rotated:
        print(f"Rotated {', '.join(result.rotated)} in {result.path}")
    elif result.created:
        print(f"Wrote {result.path} with {len(result.created)} new secrets")
    elif result.written:
        print(f"Updated {result.path}")
    else:
        print(f"{result.path} already contained the requested secrets")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
