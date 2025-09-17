from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import types
from typing import Dict, Optional

try:  # pragma: no cover - optional during tests
    from prometheus_client import start_http_server
except Exception:  # pragma: no cover - optional dependency
    start_http_server = None  # type: ignore[assignment]

try:  # pragma: no cover - optional during tests
    from opentelemetry import metrics as otel_metrics, trace as otel_trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - opentelemetry optional
    otel_metrics = None  # type: ignore
    otel_trace = None  # type: ignore
    MeterProvider = None  # type: ignore
    Resource = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    _OTEL_AVAILABLE = False

try:  # pragma: no cover - optional during tests
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
except Exception:  # pragma: no cover
    JaegerExporter = None  # type: ignore

try:  # pragma: no cover - optional during tests
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
except Exception:  # pragma: no cover
    LoggingInstrumentor = None  # type: ignore

try:  # pragma: no cover - optional during tests
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
except Exception:  # pragma: no cover
    PrometheusMetricReader = None  # type: ignore


if _OTEL_AVAILABLE:
    metrics = otel_metrics  # type: ignore[assignment]
    trace = otel_trace  # type: ignore[assignment]
    _resource = Resource.create({"service.name": "mt5"}) if Resource else None
else:
    class _NoOpCounter:
        def add(self, *args, **kwargs) -> None:  # pragma: no cover - fallback
            return None

    class _NoOpHistogram:
        def record(self, *args, **kwargs) -> None:  # pragma: no cover - fallback
            return None

    class _NoOpMeter:
        def create_counter(self, *args, **kwargs):  # pragma: no cover - fallback
            return _NoOpCounter()

        def create_histogram(self, *args, **kwargs):  # pragma: no cover - fallback
            return _NoOpHistogram()

    class _FallbackMetrics:
        def get_meter(self, name: str) -> _NoOpMeter:  # pragma: no cover - fallback
            return _NoOpMeter()

        def set_meter_provider(self, *args, **kwargs) -> None:  # pragma: no cover
            return None

    class _FallbackTrace:
        def get_tracer(self, name: str):  # pragma: no cover - fallback
            return types.SimpleNamespace(
                start_as_current_span=lambda *a, **k: contextlib.nullcontext()
            )

        def set_tracer_provider(self, *args, **kwargs) -> None:  # pragma: no cover
            return None

    metrics = _FallbackMetrics()
    trace = _FallbackTrace()
    _resource = None

_logger = logging.getLogger(__name__)
_init_lock = threading.Lock()
_telemetry_state: Dict[str, bool] = {
    "initialized": False,
    "tracing": False,
    "metrics": False,
    "logging": False,
}
_tracer_provider: Optional[TracerProvider] = None
_meter_provider: Optional[MeterProvider] = None
_prometheus_server_started = False


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def init_telemetry(force: bool = False) -> Dict[str, bool]:
    """Initialise telemetry exporters lazily.

    The configuration honours the following environment variables:

    ``ENABLE_TELEMETRY``
        Master switch to disable all telemetry exporters.
    ``ENABLE_JAEGER_EXPORTER``
        Toggle Jaeger tracing exporter setup.
    ``ENABLE_PROMETHEUS_EXPORTER``
        Toggle Prometheus metrics exporter setup.
    ``ENABLE_LOGGING_OTEL``
        Toggle OpenTelemetry logging correlation.
    """

    with _init_lock:
        if _telemetry_state["initialized"] and not force:
            return dict(_telemetry_state)

        state = {
            "initialized": True,
            "tracing": False,
            "metrics": False,
            "logging": False,
        }

        if not _OTEL_AVAILABLE:
            _logger.warning(
                "OpenTelemetry libraries unavailable; telemetry exporters disabled"
            )
            _telemetry_state.update(state)
            return dict(_telemetry_state)

        if not _env_flag("ENABLE_TELEMETRY", True):
            _logger.info("Telemetry disabled via ENABLE_TELEMETRY flag")
            _telemetry_state.update(state)
            return dict(_telemetry_state)

        # ------------------------------------------------------------------
        # Tracing configuration (Jaeger exporter)
        # ------------------------------------------------------------------
        if _env_flag("ENABLE_JAEGER_EXPORTER", True) and JaegerExporter is not None:
            try:
                exporter = JaegerExporter(
                    agent_host_name=os.getenv("JAEGER_HOST", "localhost"),
                    agent_port=int(os.getenv("JAEGER_PORT", 6831)),
                )
                provider = TracerProvider(resource=_resource)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                trace.set_tracer_provider(provider)
                global _tracer_provider
                _tracer_provider = provider
                state["tracing"] = True
            except Exception as exc:  # pragma: no cover - defensive logging
                _logger.warning("Failed to initialise Jaeger exporter: %s", exc)
        elif JaegerExporter is None and _env_flag("ENABLE_JAEGER_EXPORTER", True):
            _logger.warning(
                "Jaeger exporter is unavailable; tracing will use default provider"
            )

        # ------------------------------------------------------------------
        # Metrics configuration (Prometheus exporter)
        # ------------------------------------------------------------------
        if _env_flag("ENABLE_PROMETHEUS_EXPORTER", True):
            prom_client = sys.modules.get("prometheus_client")
            if PrometheusMetricReader and isinstance(prom_client, types.ModuleType):
                try:
                    reader = PrometheusMetricReader()
                    meter_provider = MeterProvider(
                        resource=_resource, metric_readers=[reader]
                    )
                    metrics.set_meter_provider(meter_provider)
                    global _meter_provider
                    _meter_provider = meter_provider
                    state["metrics"] = True

                    if start_http_server is not None:
                        global _prometheus_server_started
                        if not _prometheus_server_started:
                            port = int(os.getenv("PROMETHEUS_METRICS_PORT", "9464"))
                            try:
                                start_http_server(port)
                                _prometheus_server_started = True
                            except Exception as exc:  # pragma: no cover
                                _logger.warning(
                                    "Failed to start Prometheus HTTP server: %s", exc
                                )
                    else:
                        _logger.warning(
                            "prometheus_client.start_http_server unavailable;"
                            " metrics endpoint disabled"
                        )
                except Exception as exc:  # pragma: no cover - defensive logging
                    _logger.warning("Failed to initialise Prometheus exporter: %s", exc)
            else:
                if PrometheusMetricReader is None:
                    _logger.warning(
                        "Prometheus exporter unavailable; install opentelemetry-exporter-prometheus"
                    )
                else:
                    _logger.warning(
                        "prometheus_client not imported; metrics exporter not started"
                    )

        # ------------------------------------------------------------------
        # Logging instrumentation
        # ------------------------------------------------------------------
        if _env_flag("ENABLE_LOGGING_OTEL", True) and LoggingInstrumentor is not None:
            try:
                LoggingInstrumentor().instrument(set_logging_format=True)
                state["logging"] = True
            except Exception as exc:  # pragma: no cover - defensive logging
                _logger.warning("Failed to instrument logging with OpenTelemetry: %s", exc)
        elif LoggingInstrumentor is None and _env_flag("ENABLE_LOGGING_OTEL", True):
            _logger.warning(
                "OpenTelemetry logging instrumentation unavailable; install opentelemetry-instrumentation-logging"
            )

        _telemetry_state.update(state)
        return dict(_telemetry_state)


def telemetry_status() -> Dict[str, bool]:
    """Return a snapshot of the telemetry initialisation status."""

    return dict(_telemetry_state)


def get_tracer(name: str = __name__):
    return trace.get_tracer(name)


def get_meter(name: str = __name__):
    return metrics.get_meter(name)
