"""Generate Prometheus endpoint URLs for MT5 deployments.

This helper automates construction of the ``PROM_PUSH_URL`` and
``PROM_QUERY_URL`` environment variables consumed by
:mod:`analytics.metrics_aggregator`.  Operators can describe Pushgateway and
Prometheus locations (hostnames, ports, optional job labels) and the script
persists the resulting URLs to the deployment runtime environment file that the
systemd units already consume.

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
from typing import Dict, Iterable, Mapping, Sequence
from urllib.parse import quote, urlparse

DEFAULT_ENV_FILE = Path("deploy/secrets/runtime.env")
DEFAULT_PUSH_SCHEME = "http"
DEFAULT_PUSH_HOST = "pushgateway"
DEFAULT_PUSH_PORT = 9091
DEFAULT_QUERY_SCHEME = "http"
DEFAULT_QUERY_HOST = "prometheus"
DEFAULT_QUERY_PORT = 9090

MANAGED_KEYS = ("PROM_PUSH_URL", "PROM_QUERY_URL")
COMMENT_LINE = "# Prometheus endpoints managed by deployment.prometheus_endpoints.py"


@dataclass
class EnvUpdate:
    """Outcome of managing Prometheus environment variables."""

    path: Path
    values: Dict[str, str]
    created: list[str]
    updated: list[str]
    written: bool


class EnvFile:
    """Utility for reading and updating simple ``KEY=VALUE`` env files."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[str] = []
        self.values: Dict[str, str] = {}
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self.path.exists():
            self.lines = []
            self.values = {}
            return
        with self.path.open("r", encoding="utf-8") as handle:
            self.lines = [line.rstrip("\n") for line in handle]
        parsed: Dict[str, str] = {}
        for line in self.lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()
        self.values = parsed

    # ------------------------------------------------------------------
    def _index_map(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for idx, line in enumerate(self.lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in line:
                continue
            key, _ = line.split("=", 1)
            mapping[key.strip()] = idx
        return mapping

    # ------------------------------------------------------------------
    def _write(self) -> None:
        text = "\n".join(self.lines).rstrip()
        if text:
            text += "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            handle.write(text)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    # ------------------------------------------------------------------
    def _insert_new_entries(
        self,
        keys: Iterable[str],
        updates: Mapping[str, str],
        index_map: Dict[str, int],
    ) -> None:
        ordered: list[str] = [key for key in MANAGED_KEYS if key in keys]
        extras = [key for key in keys if key not in ordered]
        ordered.extend(extras)
        if not ordered:
            return

        for key in ordered:
            value = updates[key]
            line = f"{key}={value}"
            insert_at: int | None = None

            if key in MANAGED_KEYS:
                key_index = MANAGED_KEYS.index(key)
                for later_key in MANAGED_KEYS[key_index + 1 :]:
                    later_idx = index_map.get(later_key)
                    if later_idx is not None:
                        insert_at = later_idx
                        break

            if insert_at is None:
                existing_indices = [index_map[k] for k in MANAGED_KEYS if k in index_map]
                if existing_indices:
                    insert_at = max(existing_indices) + 1
                    while insert_at < len(self.lines) and self.lines[insert_at].strip():
                        insert_at += 1
                else:
                    comment_index = next(
                        (idx for idx, l in enumerate(self.lines) if l.strip() == COMMENT_LINE.strip()),
                        None,
                    )
                    if comment_index is not None:
                        insert_at = comment_index + 1
                        while insert_at < len(self.lines) and self.lines[insert_at].strip():
                            insert_at += 1
                    else:
                        insert_at = len(self.lines)
                        if insert_at and self.lines[-1].strip():
                            self.lines.append("")
                            insert_at += 1
                        self.lines.append(COMMENT_LINE)
                        insert_at += 1

            if insert_at < len(self.lines):
                self.lines.insert(insert_at, line)
            else:
                self.lines.append(line)
            self.values[key] = value
            index_map = self._index_map()

    # ------------------------------------------------------------------
    def apply(self, updates: Mapping[str, str], *, force: bool) -> EnvUpdate:
        if not updates:
            return EnvUpdate(self.path, dict(self.values), [], [], False)

        index_map = self._index_map()
        created: list[str] = []
        updated: list[str] = []

        for key, value in updates.items():
            current = self.values.get(key)
            if current is None:
                created.append(key)
                continue
            if current == value:
                continue
            idx = index_map.get(key)
            if idx is None:
                created.append(key)
                continue
            if not force:
                raise RuntimeError(
                    f"{self.path} already defines {key}; rerun with --force to overwrite it."
                )
            self.lines[idx] = f"{key}={value}"
            self.values[key] = value
            updated.append(key)

        if created:
            self._insert_new_entries(created, updates, index_map)

        written = bool(created or updated)
        if written:
            self._write()

        return EnvUpdate(self.path, dict(self.values), created, updated, written)


# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env_path = Path(args.env_file).expanduser()

    if args.skip_push and args.skip_query:
        env = EnvFile(env_path)
        if args.print_exports:
            for key in MANAGED_KEYS:
                if key in env.values:
                    print(f"export {key}='{env.values[key]}'")
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

    env = EnvFile(env_path)
    try:
        result = env.apply(values, force=args.force)
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
