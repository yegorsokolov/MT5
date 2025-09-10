import io
import json
import logging
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "logger_mod", Path(__file__).resolve().parents[1] / "utils" / "logger.py"
)
logger_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logger_mod)

get_logger = logger_mod.get_logger
TRACE_ID = logger_mod.TRACE_ID
JsonFormatter = logger_mod.JsonFormatter
TraceContextFilter = logger_mod.TraceContextFilter

def _make_logger(name: str):
    logger = get_logger(name)
    logger.setLevel(logging.INFO)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(TraceContextFilter())
    logger.handlers = [handler]
    return logger, stream

def test_json_log_basic_fields():
    logger, stream = _make_logger("test.module")
    logger.info("hello")
    data = json.loads(stream.getvalue().strip())
    assert data["level"] == "INFO"
    assert data["module"] == "test.module"
    assert data["message"] == "hello"
    assert "timestamp" in data
    assert "trace_id" not in data

def test_json_log_with_trace_id():
    logger, stream = _make_logger("test.module")
    token = TRACE_ID.set("abc123")
    try:
        logger.warning("warn")
    finally:
        TRACE_ID.reset(token)
    data = json.loads(stream.getvalue().strip())
    assert data["trace_id"] == "abc123"
    assert data["level"] == "WARNING"
    assert data["module"] == "test.module"
    assert data["message"] == "warn"
    assert "timestamp" in data
