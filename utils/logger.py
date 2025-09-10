import logging
import json
import contextvars
from datetime import datetime, timezone

TRACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


class JsonFormatter(logging.Formatter):
    """Format logs as JSON with standard fields."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        trace_id = getattr(record, "trace_id", None)
        if trace_id:
            log_record["trace_id"] = trace_id
        return json.dumps(log_record)


class TraceContextFilter(logging.Filter):
    """Inject trace_id from contextvars into log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple
        trace_id = TRACE_ID.get()
        if trace_id:
            record.trace_id = trace_id
        return True


_CONTEXT_FILTER = TraceContextFilter()


def get_logger(name: str) -> logging.Logger:
    """Return a logger that emits JSON and injects trace IDs."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler.addFilter(_CONTEXT_FILTER)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_trace_id(trace_id: str) -> contextvars.Token:
    """Set the correlation/trace id for the current context."""
    return TRACE_ID.set(trace_id)


def clear_trace_id(token: contextvars.Token) -> None:
    """Reset the trace id to its previous value."""
    TRACE_ID.reset(token)


__all__ = [
    "get_logger",
    "set_trace_id",
    "clear_trace_id",
    "TRACE_ID",
    "JsonFormatter",
    "TraceContextFilter",
]
