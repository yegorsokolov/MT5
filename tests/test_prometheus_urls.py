from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deployment import prometheus_urls


def read_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def test_prometheus_urls_creates_defaults(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    result = prometheus_urls.ensure_prometheus_urls(env_file)

    assert result.written is True
    assert result.created == [prometheus_urls.PUSH_KEY, prometheus_urls.QUERY_KEY]
    assert result.values[prometheus_urls.PUSH_KEY] == "http://localhost:9091"
    assert result.values[prometheus_urls.QUERY_KEY] == "http://localhost:9090"


def test_prometheus_urls_idempotent(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    first = prometheus_urls.ensure_prometheus_urls(env_file)
    second = prometheus_urls.ensure_prometheus_urls(env_file)

    assert first.written is True
    assert second.written is False
    assert second.created == []
    assert second.updated == []


def test_prometheus_urls_requires_force_for_overwrite(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    prometheus_urls.ensure_prometheus_urls(env_file)

    with pytest.raises(RuntimeError) as excinfo:
        prometheus_urls.ensure_prometheus_urls(
            env_file,
            push_host="example.com",
        )

    assert prometheus_urls.PUSH_KEY in str(excinfo.value)


def test_prometheus_urls_force_overwrite(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    prometheus_urls.ensure_prometheus_urls(env_file)

    result = prometheus_urls.ensure_prometheus_urls(
        env_file,
        push_host="push.example",
        query_host="prom.example",
        force=True,
    )

    assert set(result.updated) == {prometheus_urls.PUSH_KEY, prometheus_urls.QUERY_KEY}
    assert result.values[prometheus_urls.PUSH_KEY] == "http://push.example:9091"
    assert result.values[prometheus_urls.QUERY_KEY] == "http://prom.example:9090"


def test_prometheus_urls_with_job_and_instance(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    result = prometheus_urls.ensure_prometheus_urls(
        env_file,
        push_job="mt5bot",
        push_instance="primary",
    )

    expected = "http://localhost:9091/metrics/job/mt5bot/instance/primary"
    assert result.values[prometheus_urls.PUSH_KEY] == expected


def test_prometheus_urls_disable_entries(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    result = prometheus_urls.ensure_prometheus_urls(
        env_file,
        disable_push=True,
        disable_query=True,
    )

    assert result.values[prometheus_urls.PUSH_KEY] == ""
    assert result.values[prometheus_urls.QUERY_KEY] == ""


def test_prometheus_urls_preserves_existing_entries(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    env_file.write_text("CONFIG_ADMIN_KEY=abc123\n")

    result = prometheus_urls.ensure_prometheus_urls(env_file, force=True)
    persisted = read_env(env_file)

    assert result.values["CONFIG_ADMIN_KEY"] == "abc123"
    assert persisted["CONFIG_ADMIN_KEY"] == "abc123"
