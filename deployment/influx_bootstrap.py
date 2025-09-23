"""Automate InfluxDB initial setup and bucket token provisioning.

The script can either perform the one-time ``/api/v2/setup`` bootstrap when a
fresh InfluxDB instance is detected or create the required bucket and token
against an already-initialised deployment.  It is intended to be executed by
deployment automation so operators no longer have to run ``influx setup``
manually and copy credentials around.

Example usage (non-interactive):

.. code-block:: bash

    python -m deployment.influx_bootstrap \\
        --url http://influxdb:8086 \\
        --org trading \\
        --bucket mt5_metrics \\
        --admin-token "$INFLUX_ADMIN" \\
        --env-file /run/secrets/influxdb.env \\
        --print-exports

The resulting token is restricted to the requested bucket and can safely be
stored wherever you keep runtime secrets (environment file, Vault, Docker
secrets, etc.).  By default the script refuses to overwrite existing secret
files unless ``--force`` is provided so that it can be executed repeatedly
without accidentally rotating credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import argparse
import json
import os
import re
import secrets
import string
import sys

import requests

DEFAULT_USERNAME = "mt5-automation"
DEFAULT_DESCRIPTION = "MT5 automation bucket access"
DEFAULT_TIMEOUT = 15


@dataclass
class SetupResult:
    org: Dict[str, object]
    bucket: Dict[str, object]
    admin_token: str
    admin_password: str | None = None


class BootstrapError(RuntimeError):
    """Raised when bootstrapping cannot proceed safely."""


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision InfluxDB org, bucket and automation token")
    parser.add_argument("--url", required=True, help="Base URL of the InfluxDB instance, e.g. http://localhost:8086")
    parser.add_argument("--org", required=True, help="Organisation to create or reuse")
    parser.add_argument("--bucket", required=True, help="Bucket to create or reuse")
    parser.add_argument(
        "--username",
        default=DEFAULT_USERNAME,
        help="Admin username used when running the one-time /api/v2/setup bootstrap",
    )
    parser.add_argument(
        "--password",
        help="Admin password for initial setup. A random password is generated when omitted.",
    )
    parser.add_argument(
        "--admin-token",
        help="Existing admin token. Required if the instance was already initialised.",
    )
    parser.add_argument(
        "--retention",
        help="Retention period for the bucket (accepts seconds or values like 72h, 7d).",
    )
    parser.add_argument(
        "--token-description",
        default=DEFAULT_DESCRIPTION,
        help="Description recorded on the generated bucket token.",
    )
    parser.add_argument(
        "--env-file",
        help="Optional path to write INFLUXDB_* environment variables for the runtime service.",
    )
    parser.add_argument(
        "--print-exports",
        action="store_true",
        help="Emit 'export KEY=VALUE' commands for shell usage in addition to file output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the env file even when it already exists.",
    )
    parser.add_argument(
        "--rotate-token",
        action="store_true",
        help="Always mint a fresh automation token even if one with the same description already exists.",
    )
    parser.add_argument(
        "--store-admin-secret",
        action="store_true",
        help="Also persist the admin API token to the env file as INFLUXDB_ADMIN_TOKEN.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds for talking to the InfluxDB API (default: 15).",
    )
    return parser.parse_args(argv)


def random_password(length: int = 32) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def parse_retention(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    if value.isdigit():
        return int(value)
    match = re.fullmatch(r"(\d+)([smhdw])", value.strip().lower())
    if not match:
        raise BootstrapError(
            "Retention must be specified as raw seconds or suffixed with s, m, h, d or w (e.g. 72h)."
        )
    amount = int(match.group(1))
    unit = match.group(2)
    multiplier = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }[unit]
    return amount * multiplier


def normalise_base_url(url: str) -> str:
    """Return the API base URL without a trailing slash."""

    return url[:-1] if url.endswith("/") else url


def setup_allowed(session: requests.Session, base_url: str, timeout: int) -> bool:
    resp = session.get(f"{base_url}/api/v2/setup", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return bool(data.get("allowed", False))


def initial_setup(
    session: requests.Session,
    *,
    base_url: str,
    org: str,
    bucket: str,
    username: str,
    password: Optional[str],
    retention: Optional[int],
    timeout: int,
) -> SetupResult:
    password_value = password or random_password()
    payload: Dict[str, object] = {
        "org": org,
        "bucket": bucket,
        "username": username,
        "password": password_value,
    }
    if retention is not None:
        payload["retentionPeriodSeconds"] = retention
    resp = session.post(f"{base_url}/api/v2/setup", json=payload, timeout=timeout)
    if resp.status_code == 412:  # already set up
        raise BootstrapError("InfluxDB reports it has already been initialised; supply --admin-token instead.")
    resp.raise_for_status()
    data = resp.json()
    auth = data.get("auth", {})
    token = auth.get("token")
    if not token:
        raise BootstrapError("Initial setup succeeded but no admin token was returned by the API.")
    return SetupResult(
        org=data.get("org", {}),
        bucket=data.get("bucket", {}),
        admin_token=token,
        admin_password=password_value,
    )


def api_headers(admin_token: str) -> Dict[str, str]:
    return {"Authorization": f"Token {admin_token}", "Content-Type": "application/json"}


def find_org(session: requests.Session, base_url: str, admin_token: str, org: str, timeout: int) -> Dict[str, object]:
    resp = session.get(
        f"{base_url}/api/v2/orgs",
        params={"org": org, "limit": 1},
        headers=api_headers(admin_token),
        timeout=timeout,
    )
    resp.raise_for_status()
    items = resp.json().get("orgs", [])
    if not items:
        resp = session.post(
            f"{base_url}/api/v2/orgs",
            headers=api_headers(admin_token),
            data=json.dumps({"name": org}),
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    return items[0]


def find_bucket(
    session: requests.Session,
    base_url: str,
    admin_token: str,
    org_id: str,
    bucket: str,
    retention: Optional[int],
    timeout: int,
) -> Dict[str, object]:
    resp = session.get(
        f"{base_url}/api/v2/buckets",
        params={"name": bucket, "orgID": org_id, "limit": 1},
        headers=api_headers(admin_token),
        timeout=timeout,
    )
    resp.raise_for_status()
    items = resp.json().get("buckets", [])
    if items:
        return items[0]

    body: Dict[str, object] = {"name": bucket, "orgID": org_id}
    if retention is not None and retention > 0:
        body["retentionRules"] = [{"type": "expire", "everySeconds": retention}]
    resp = session.post(
        f"{base_url}/api/v2/buckets",
        headers=api_headers(admin_token),
        data=json.dumps(body),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def find_existing_token(
    session: requests.Session,
    base_url: str,
    admin_token: str,
    description: str,
    bucket_id: str,
    timeout: int,
) -> Optional[str]:
    resp = session.get(
        f"{base_url}/api/v2/authorizations",
        params={"limit": 100},
        headers=api_headers(admin_token),
        timeout=timeout,
    )
    resp.raise_for_status()
    for auth in resp.json().get("authorizations", []):
        if auth.get("description") != description:
            continue
        permissions = auth.get("permissions", [])
        for perm in permissions:
            resource = perm.get("resource", {})
            if resource.get("type") == "buckets" and resource.get("id") == bucket_id:
                return auth.get("token")
    return None


def create_bucket_token(
    session: requests.Session,
    base_url: str,
    admin_token: str,
    org_id: str,
    bucket_id: str,
    description: str,
    *,
    rotate: bool,
    timeout: int,
) -> str:
    if not rotate:
        existing = find_existing_token(session, base_url, admin_token, description, bucket_id, timeout)
        if existing:
            return existing

    body = {
        "orgID": org_id,
        "description": description,
        "permissions": [
            {
                "action": "read",
                "resource": {"type": "buckets", "orgID": org_id, "id": bucket_id},
            },
            {
                "action": "write",
                "resource": {"type": "buckets", "orgID": org_id, "id": bucket_id},
            },
        ],
    }
    resp = session.post(
        f"{base_url}/api/v2/authorizations",
        headers=api_headers(admin_token),
        data=json.dumps(body),
        timeout=timeout,
    )
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise BootstrapError("Failed to create automation token – response did not include a token string.")
    return token


def read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            values[key] = raw_value
    return values


def write_env_file(path: Path, values: Dict[str, str], *, force: bool) -> None:
    if path.exists() and not force:
        current = read_env_file(path)
        if all(current.get(key) == val for key, val in values.items()):
            return
        raise BootstrapError(f"Secret file {path} already exists – use --force to overwrite or rotate the token.")

    existing: Dict[str, str] = {}
    if path.exists():
        existing = read_env_file(path)
    existing.update(values)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, val in sorted(existing.items()):
            handle.write(f"{key}={val}\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    session = requests.Session()
    base_url = normalise_base_url(args.url)
    retention_seconds = parse_retention(args.retention)
    env_path = Path(args.env_file).expanduser() if args.env_file else None
    admin_token_env = os.getenv("INFLUXDB_ADMIN_TOKEN")
    if not args.admin_token and admin_token_env:
        args.admin_token = admin_token_env

    if env_path and env_path.exists() and not args.force and not args.rotate_token:
        existing = read_env_file(env_path)
        if all(
            existing.get(key) == value
            for key, value in {
                "INFLUXDB_URL": base_url,
                "INFLUXDB_ORG": args.org,
                "INFLUXDB_BUCKET": args.bucket,
            }.items()
        ) and "INFLUXDB_TOKEN" in existing:
            print(f"Secrets file {env_path} already populated; skipping bootstrap.")
            if args.print_exports:
                for key in ("INFLUXDB_URL", "INFLUXDB_ORG", "INFLUXDB_BUCKET", "INFLUXDB_TOKEN"):
                    value = existing.get(key)
                    if value is not None:
                        print(f"export {key}='{value}'")
            return 0

    try:
        allowed = setup_allowed(session, base_url, args.timeout)
    except requests.HTTPError as exc:  # 404 indicates already initialised on older versions
        if exc.response is not None and exc.response.status_code == 404:
            allowed = False
        else:
            raise

    setup_result: Optional[SetupResult] = None
    admin_token = args.admin_token
    admin_password: Optional[str] = None

    if allowed:
        setup_result = initial_setup(
            session,
            base_url=base_url,
            org=args.org,
            bucket=args.bucket,
            username=args.username,
            password=args.password,
            retention=retention_seconds,
            timeout=args.timeout,
        )
        admin_token = setup_result.admin_token
        admin_password = setup_result.admin_password
        org_info = setup_result.org
        bucket_info = setup_result.bucket
    else:
        if not admin_token:
            raise BootstrapError(
                "InfluxDB is already set up. Provide an admin token via --admin-token or the INFLUXDB_ADMIN_TOKEN environment variable."
            )
        org_info = find_org(session, base_url, admin_token, args.org, args.timeout)
        org_id = org_info.get("id")
        if not isinstance(org_id, str):
            raise BootstrapError("Unable to determine organisation ID from InfluxDB response.")
        bucket_info = find_bucket(
            session,
            base_url,
            admin_token,
            org_id,
            args.bucket,
            retention_seconds,
            args.timeout,
        )

    org_id = org_info.get("id") if isinstance(org_info, dict) else None
    bucket_id = bucket_info.get("id") if isinstance(bucket_info, dict) else None
    if not isinstance(org_id, str) or not isinstance(bucket_id, str):
        raise BootstrapError("InfluxDB did not return valid IDs for the org or bucket.")

    assert admin_token is not None  # for type-checkers

    token = create_bucket_token(
        session,
        base_url,
        admin_token,
        org_id,
        bucket_id,
        args.token_description,
        rotate=args.rotate_token,
        timeout=args.timeout,
    )

    exports = {
        "INFLUXDB_URL": base_url,
        "INFLUXDB_ORG": args.org,
        "INFLUXDB_BUCKET": args.bucket,
        "INFLUXDB_TOKEN": token,
    }
    if args.store_admin_secret and admin_token:
        exports["INFLUXDB_ADMIN_TOKEN"] = admin_token

    if env_path:
        write_env_file(env_path, exports, force=args.force or args.rotate_token)

    if args.print_exports:
        for key, value in exports.items():
            print(f"export {key}='{value}'")

    if setup_result:
        print("InfluxDB initialised successfully.")
        print(f"  Admin username: {args.username}")
        if admin_password:
            print(f"  Admin password: {admin_password}")
        print("Remember to store the admin credentials securely; they are not written to disk by default.")
    else:
        print("InfluxDB bucket and token ensured successfully.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BootstrapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as exc:
        print(f"error: failed to communicate with InfluxDB: {exc}", file=sys.stderr)
        sys.exit(2)
