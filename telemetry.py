import os
try:  # pragma: no cover - optional during tests
    from prometheus_client import start_http_server
except Exception:  # pragma: no cover
    def start_http_server(*args, **kwargs):
        pass
from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.logging import LoggingInstrumentor
import sys
import types
try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
except Exception:  # pragma: no cover
    PrometheusMetricReader = None  # type: ignore

_resource = Resource.create({"service.name": "mt5"})

# Tracing setup
_jaeger_exporter = JaegerExporter(
    agent_host_name=os.getenv("JAEGER_HOST", "localhost"),
    agent_port=int(os.getenv("JAEGER_PORT", 6831)),
)
_tracer_provider = TracerProvider(resource=_resource)
_tracer_provider.add_span_processor(BatchSpanProcessor(_jaeger_exporter))
trace.set_tracer_provider(_tracer_provider)

# Metrics setup
_metric_readers = []
_prom_client = sys.modules.get("prometheus_client")
if PrometheusMetricReader and isinstance(_prom_client, types.ModuleType):
    reader = PrometheusMetricReader()
    _metric_readers.append(reader)
    _start_port = int(os.getenv("PROMETHEUS_METRICS_PORT", "9464"))
    try:
        start_http_server(_start_port)
    except Exception:
        pass
_meter_provider = MeterProvider(resource=_resource, metric_readers=_metric_readers)
metrics.set_meter_provider(_meter_provider)

# Logging correlation
LoggingInstrumentor().instrument(set_logging_format=True)


def get_tracer(name: str = __name__):
    return trace.get_tracer(name)


def get_meter(name: str = __name__):
    return metrics.get_meter(name)
