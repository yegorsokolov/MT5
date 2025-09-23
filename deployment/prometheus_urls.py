"""Generate Prometheus endpoint URLs for runtime services.

This helper mirrors :mod:`deployment.runtime_secrets` by materialising
configuration during provisioning so operators no longer have to edit ``.env``
manually.  It stores the Pushgateway endpoint used by analytics jobs and the
Prometheus HTTP API base URL alongside the other generated values under
``deploy/secrets/``.

Example usage::

    python -m deployment.prometheus_urls \\
        --push-host pushgateway \\
        --query-host prometheus \\
        --env-file deploy/secrets/runtime.env \\
        --print-exports

The script is idempotent: rerunning it reuses existing values unless ``--force``
is supplied, allowing unattended installers to call it repeatedly without
accidentally overwriting customised endpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence
from urllib.parse import urlunparse
import sys

DEFAULT_ENV_FILE = Path("deploy/secrets/runtime.env")
DEFAULT_PUSH_SCHEME = "http"
DEFAULT_PUSH_HOST = "localhost"
DEFAULT_PUSH_PORT = 9091
DEFAULT_QUERY_SCHEME = "http"
DEFAULT_QUERY_HOST = "localhost"
DEFAULT_QUERY_PORT = 9090

PUSH_KEY = "PROM_PUSH_URL"
QUERY_KEY = "PROM_QUERY_URL"


@dataclass
class PrometheusUrlResult:
    """Outcome of ensuring Prometheus URLs exist."""

    path: Path
    values: Dict[str, str]
    created: list[str]
    updated: list[str]
    written: bool


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


def _normalise_path(value: str | None) -> str:
    if not value:
        return ""
    stripped = value.strip()
    if not stripped:
        return ""
    return stripped if stripped.startswith("/") else f"/{stripped}"


def _compose_push_path(
    base_path: str | None,
    job: str | None,
    instance: str | None,
) -> str:
    base = _normalise_path(base_path)
    job_name = job.strip() if job else ""
    instance_name = instance.strip() if instance else ""
    job_path = ""
    if job_name:
        job_path = f"/metrics/job/{job_name}"
        if instance_name:
            job_path += f"/instance/{instance_name}"
    if base and job_path:
        if base.endswith("/"):
            base = base.rstrip("/")
        return f"{base}{job_path}"
    return job_path or base


def _compose_url(scheme: str, host: str, port: int | None, path: str) -> str:
    scheme_value = (scheme or DEFAULT_PUSH_SCHEME).strip().lower() or DEFAULT_PUSH_SCHEME
    host_value = (host or "").strip()
    if not host_value:
        raise RuntimeError("Host must not be empty when generating Prometheus URLs")
    clean_port = port if port and port > 0 else None
    netloc = host_value if clean_port is None else f"{host_value}:{clean_port}"
    clean_path = path or ""
    return urlunparse((scheme_value, netloc, clean_path, "", "", ""))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Prometheus endpoint URLs")
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to the env file receiving PROM_PUSH_URL/PROM_QUERY_URL (default: deploy/secrets/runtime.env)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing values without prompting.",
    )
    parser.add_argument(
        "--push-url",
        help="Explicit Pushgateway URL. Overrides host/port/job parameters when provided.",
    )
    parser.add_argument(
        "--push-scheme",
        default=DEFAULT_PUSH_SCHEME,
        help="Scheme to use for the Pushgateway URL (default: http).",
    )
    parser.add_argument(
        "--push-host",
        default=DEFAULT_PUSH_HOST,
        help="Hostname for the Pushgateway URL (default: localhost).",
    )
    parser.add_argument(
        "--push-port",
        type=int,
        default=DEFAULT_PUSH_PORT,
        help="Port for the Pushgateway URL. Use 0 to omit the port (default: 9091).",
    )
    parser.add_argument(
        "--push-path",
        help="Additional path prefix for the Pushgateway URL (prepended before any job/instance segments).",
    )
    parser.add_argument(
        "--push-job",
        help="Pushgateway job label appended as /metrics/job/<job>.",
    )
    parser.add_argument(
        "--push-instance",
        help="Pushgateway instance label appended as /instance/<instance>.",
    )
    parser.add_argument(
        "--disable-push",
        action="store_true",
        help="Write an empty PROM_PUSH_URL entry to disable push-style metrics.",
    )
    parser.add_argument(
        "--query-url",
        help="Explicit Prometheus HTTP API base URL. Overrides host/port/path parameters when provided.",
    )
    parser.add_argument(
        "--query-scheme",
        default=DEFAULT_QUERY_SCHEME,
        help="Scheme to use for the Prometheus API URL (default: http).",
    )
    parser.add_argument(
        "--query-host",
        default=DEFAULT_QUERY_HOST,
        help="Hostname for the Prometheus API URL (default: localhost).",
    )
    parser.add_argument(
        "--query-port",
        type=int,
        default=DEFAULT_QUERY_PORT,
        help="Port for the Prometheus API URL. Use 0 to omit the port (default: 9090).",
    )
    parser.add_argument(
        "--query-path",
        help="Path appended to the Prometheus API URL (e.g. /prometheus).",
    )
    parser.add_argument(
        "--disable-query",
        action="store_true",
        help="Write an empty PROM_QUERY_URL entry to disable Prometheus API lookups.",
    )
    parser.add_argument(
        "--print-exports",
        action="store_true",
        help="Print export commands for shell usage.",
    )
    return parser.parse_args(argv)


def _apply_changes(
    existing: Mapping[str, str],
    managed: MutableMapping[str, str],
    *,
    force: bool,
) -> tuple[list[str], list[str]]:
    created: list[str] = []
    updated: list[str] = []
    for key, value in managed.items():
        current = existing.get(key)
        if current is None:
            created.append(key)
        elif current != value:
            if not force:
                raise RuntimeError(
                    f"Environment value {key} already exists – rerun with --force to overwrite it."
                )
            updated.append(key)
    return created, updated


def _write_env_file(
    path: Path,
    *,
    managed: Mapping[str, str],
    existing: Mapping[str, str],
    force: bool,
) -> bool:
    merged: Dict[str, str] = dict(existing)
    changed = False
    for key, value in managed.items():
        current = merged.get(key)
        if current == value:
            continue
        if current is not None and not force:
            raise RuntimeError(
                f"Environment value {key} already exists – rerun with --force to overwrite it."
            )
        merged[key] = value
        changed = True
    if not changed and all(key in merged for key in managed):
        return False

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Prometheus endpoints generated by deployment.prometheus_urls",
        f"# Updated {timestamp}",
        "#",
        "# PROM_PUSH_URL  – Endpoint accepting text exposition format (e.g. Pushgateway).",
        "# PROM_QUERY_URL – Base URL for Prometheus HTTP API queries.",
        "",
    ]
    for key in (PUSH_KEY, QUERY_KEY):
        if key in merged:
            lines.append(f"{key}={merged[key]}")
    lines.append("")

    preserved = [
        (key, merged[key])
        for key in sorted(merged)
        if key not in {PUSH_KEY, QUERY_KEY} and not key.startswith("#")
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


def ensure_prometheus_urls(
    env_path: Path,
    *,
    push_url: str | None = None,
    push_scheme: str = DEFAULT_PUSH_SCHEME,
    push_host: str = DEFAULT_PUSH_HOST,
    push_port: int | None = DEFAULT_PUSH_PORT,
    push_path: str | None = None,
    push_job: str | None = None,
    push_instance: str | None = None,
    disable_push: bool = False,
    query_url: str | None = None,
    query_scheme: str = DEFAULT_QUERY_SCHEME,
    query_host: str = DEFAULT_QUERY_HOST,
    query_port: int | None = DEFAULT_QUERY_PORT,
    query_path: str | None = None,
    disable_query: bool = False,
    force: bool = False,
) -> PrometheusUrlResult:
    existing = _read_env_file(env_path)

    managed: Dict[str, str] = {}

    if disable_push:
        target_push = ""
    elif push_url:
        target_push = push_url.strip()
    else:
        path = _compose_push_path(push_path, push_job, push_instance)
        target_push = _compose_url(push_scheme, push_host, push_port, path)
    managed[PUSH_KEY] = target_push

    if disable_query:
        target_query = ""
    elif query_url:
        target_query = query_url.strip()
    else:
        query_path_value = _normalise_path(query_path)
        target_query = _compose_url(query_scheme, query_host, query_port, query_path_value)
    managed[QUERY_KEY] = target_query

    created, updated = _apply_changes(existing, managed, force=force)
    write_force = force or bool(updated)

    written = _write_env_file(
        env_path,
        managed=managed,
        existing=existing,
        force=write_force,
    )

    final_values = dict(existing)
    final_values.update(managed)
    if written and not final_values:
        final_values = _read_env_file(env_path)

    return PrometheusUrlResult(
        path=env_path,
        values=final_values,
        created=created,
        updated=updated,
        written=written,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env_path = Path(args.env_file).expanduser()
    try:
        result = ensure_prometheus_urls(
            env_path,
            push_url=args.push_url,
            push_scheme=args.push_scheme,
            push_host=args.push_host,
            push_port=args.push_port,
            push_path=args.push_path,
            push_job=args.push_job,
            push_instance=args.push_instance,
            disable_push=args.disable_push,
            query_url=args.query_url,
            query_scheme=args.query_scheme,
            query_host=args.query_host,
            query_port=args.query_port,
            query_path=args.query_path,
            disable_query=args.disable_query,
            force=args.force,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.print_exports:
        for key in (PUSH_KEY, QUERY_KEY):
            value = result.values.get(key, "")
            print(f"export {key}='{value}'")

    if result.updated:
        print(
            f"Updated {result.path} (overwrote {', '.join(result.updated)})"
        )
    elif result.created:
        print(f"Wrote {result.path} with {len(result.created)} new entries")
    elif result.written:
        print(f"Updated {result.path}")
    else:
        print(f"{result.path} already contained the requested Prometheus URLs")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
