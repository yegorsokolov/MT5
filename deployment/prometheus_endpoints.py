"""Generate Prometheus endpoint URLs for MT5 deployments.

This helper automates construction of the ``PROM_PUSH_URL`` and
``PROM_QUERY_URL`` environment variables consumed by
:mod:`analytics.metrics_aggregator`.  Operators can describe Pushgateway and
Prometheus locations (hostnames, ports, optional job labels) and the script
persists the resulting URLs to the deployment runtime environment file that the
systemd units already consume.

Example usage::

    python -m deployment.prometheus_endpoints \\
        --push-host pushgateway \\
        --query-host prometheus \\
        --env-file deploy/secrets/runtime.env \\
        --print-exports

The command is idempotent: existing values are reused unless ``--force`` is
specified. Paths for custom reverse proxies or Pushgateway job labels can be
provided with the relevant CLI flags.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Dict, Mapping, Sequence
from urllib.parse import quote, urlparse

DEFAULT_ENV_FILE = Path("deploy/secrets/runtime.env")
DEFAULT_PUSH_SCHEME = "http"
DEFAULT_PUSH_HOST = "pushgateway"
DEFAULT_PUSH_PORT = 9091
DEFAULT_QUERY_SCHEME = "http"
DEFAULT_QUERY_HOST = "prometheus"
DEFAULT_QUERY_PORT = 9090

MANAGED_KEYS = ("PROM_PUSH_URL", "PROM_QUERY_URL")


@dataclass
class PrometheusEnvResult:
    """Outcome of managing Prometheus environment variables."""

    path: Path
    values: Dict[str, str]
    created: list[str]
    updated: list[str]
    written: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PROM_PUSH_URL and PROM_QUERY_URL environment entries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to the environment file that should store the Prometheus URLs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing entries when they differ from the generated values.",
    )
    parser.add_argument(
        "--print-exports",
        action="store_true",
        help="Emit 'export KEY=VALUE' lines for the managed variables.",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Do not manage PROM_PUSH_URL in the target environment file.",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Do not manage PROM_QUERY_URL in the target environment file.",
    )

    push_group = parser.add_argument_group("Pushgateway discovery")
    push_group.add_argument(
        "--push-url",
        help="Explicit Pushgateway URL. When provided all other --push-* options are ignored.",
    )
    push_group.add_argument(
        "--push-scheme",
        default=DEFAULT_PUSH_SCHEME,
        help="Scheme to use when building the Pushgateway URL.",
    )
    push_group.add_argument(
        "--push-host",
        default=DEFAULT_PUSH_HOST,
        help="Hostname for the Pushgateway service.",
    )
    push_group.add_argument(
        "--push-port",
        type=int,
        default=DEFAULT_PUSH_PORT,
        help="Port for the Pushgateway service. Use 0 to omit the port from the URL.",
    )
    push_group.add_argument(
        "--push-path",
        default="",
        help="Additional path appended to the Pushgateway URL (for reverse proxies).",
    )
    push_group.add_argument(
        "--push-job",
        help="Optional Pushgateway job label inserted as /metrics/job/<job>.",
    )
    push_group.add_argument(
        "--push-instance",
        help="Optional Pushgateway instance label appended after the job segment.",
    )

    query_group = parser.add_argument_group("Prometheus API discovery")
    query_group.add_argument(
        "--query-url",
        help="Explicit Prometheus base URL. When provided all other --query-* options are ignored.",
    )
    query_group.add_argument(
        "--query-scheme",
        default=DEFAULT_QUERY_SCHEME,
        help="Scheme to use when building the Prometheus base URL.",
    )
    query_group.add_argument(
        "--query-host",
        default=DEFAULT_QUERY_HOST,
        help="Hostname for the Prometheus server.",
    )
    query_group.add_argument(
        "--query-port",
        type=int,
        default=DEFAULT_QUERY_PORT,
        help="Port for the Prometheus server. Use 0 to omit the port from the URL.",
    )
    query_group.add_argument(
        "--query-path",
        default="",
        help="Additional path appended to the Prometheus base URL (e.g. /prometheus).",
    )

    return parser.parse_args(argv)


def _normalise_scheme(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Scheme cannot be empty.")
    if "://" in cleaned:
        cleaned = cleaned.split("://", 1)[0]
    return cleaned.lower()


def _normalise_host(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Host cannot be empty.")
    return cleaned


def _normalise_path(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.strip()
    if not cleaned:
        return ""
    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned
    return cleaned


def _normalise_port(port: int | None) -> int | None:
    if port is None or port <= 0:
        return None
    return port


def _compose_push_path(job: str | None, instance: str | None, extra: str) -> str:
    parts: list[str] = []
    job_value = (job or "").strip()
    if job_value:
        segment = f"/metrics/job/{quote(job_value, safe='')}"
        instance_value = (instance or "").strip()
        if instance_value:
            segment += f"/instance/{quote(instance_value, safe='')}"
        parts.append(segment)
    extra_path = _normalise_path(extra)
    if extra_path:
        parts.append(extra_path)
    return "".join(parts)


def _build_url(*, scheme: str, host: str, port: int | None, path: str) -> str:
    scheme_value = _normalise_scheme(scheme)
    host_value = _normalise_host(host)
    netloc = host_value if port is None else f"{host_value}:{port}"
    path_value = path or ""
    if path_value and not path_value.startswith("/"):
        path_value = "/" + path_value
    return f"{scheme_value}://{netloc}{path_value}"


def _validate_explicit_url(url: str, label: str) -> str:
    cleaned = url.strip()
    if not cleaned:
        raise ValueError(f"{label} URL cannot be empty.")
    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"{label} URL must include a scheme and host component.")
    return cleaned


def determine_push_url(args: argparse.Namespace) -> str:
    if args.skip_push:
        raise RuntimeError("Push URL requested while --skip-push is active.")
    if args.push_url:
        return _validate_explicit_url(args.push_url, "Pushgateway")
    path = _compose_push_path(args.push_job, args.push_instance, args.push_path)
    port = _normalise_port(args.push_port)
    return _build_url(
        scheme=args.push_scheme,
        host=args.push_host,
        port=port,
        path=path,
    )


def determine_query_url(args: argparse.Namespace) -> str:
    if args.skip_query:
        raise RuntimeError("Query URL requested while --skip-query is active.")
    if args.query_url:
        return _validate_explicit_url(args.query_url, "Prometheus")
    path = _normalise_path(args.query_path)
    port = _normalise_port(args.query_port)
    return _build_url(
        scheme=args.query_scheme,
        host=args.query_host,
        port=port,
        path=path,
    )


def _read_env(path: Path) -> Dict[str, str]:
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


def _load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle.readlines()]


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    content = "\n".join(lines).rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def ensure_prometheus_env(
    env_path: Path,
    values: Mapping[str, str],
    *,
    force: bool,
) -> PrometheusEnvResult:
    existing = _read_env(env_path)
    if not values:
        return PrometheusEnvResult(env_path, existing, [], [], False)

    created: list[str] = []
    updated: list[str] = []

    for key, value in values.items():
        current = existing.get(key)
        if current == value:
            continue
        if current is None:
            created.append(key)
        else:
            if not force:
                raise RuntimeError(
                    f"{env_path} already defines {key}; rerun with --force to overwrite it."
                )
            updated.append(key)

    if not created and not updated:
        final_values = dict(existing)
        final_values.update(values)
        return PrometheusEnvResult(env_path, final_values, [], [], False)

    lines = _load_lines(env_path)
    managed_keys = set(values)

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _ = stripped.split("=", 1)
        key = key.strip()
        if key in managed_keys:
            lines[idx] = f"{key}={values[key]}"

    new_entries = [
        f"{key}={values[key]}"
        for key in MANAGED_KEYS
        if key in created and key in values
    ]
    if new_entries:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(new_entries)

    _write_lines(env_path, lines)

    final_values = dict(existing)
    final_values.update({key: values[key] for key in managed_keys})
    return PrometheusEnvResult(
        path=env_path,
        values=final_values,
        created=created,
        updated=updated,
        written=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env_path = Path(args.env_file).expanduser()

    if args.skip_push and args.skip_query:
        existing = _read_env(env_path)
        if args.print_exports:
            for key in MANAGED_KEYS:
                if key in existing:
                    print(f"export {key}='{existing[key]}'")
        print("No Prometheus endpoints selected; nothing to do.")
        return 0

    values: Dict[str, str] = {}

    try:
        if not args.skip_push:
            values["PROM_PUSH_URL"] = determine_push_url(args)
        if not args.skip_query:
            values["PROM_QUERY_URL"] = determine_query_url(args)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        result = ensure_prometheus_env(env_path, values, force=args.force)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.print_exports:
        for key in MANAGED_KEYS:
            if key in result.values:
                print(f"export {key}='{result.values[key]}'")

    if result.updated:
        print(f"Updated {', '.join(result.updated)} in {result.path}")
    elif result.created:
        print(f"Wrote {result.path} with {len(result.created)} new values")
    elif result.written:
        print(f"Updated {result.path}")
    else:
        print(f"{result.path} already contained the requested values")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
