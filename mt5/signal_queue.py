"""Signal queue built on the shared :mod:`services.message_bus`."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import threading
from typing import Any, AsyncGenerator, Dict
import pandas as pd
import numpy as np

from services.message_bus import Topics, get_message_bus, MessageBus

from analysis import pipeline_anomaly
from event_store.event_writer import record as record_event
from models.conformal import evaluate_coverage

try:  # pragma: no cover - optional dependency
    from models import residual_stacker as _residual_stacker
except Exception:  # pragma: no cover - residual stacking is optional
    _residual_stacker = None

_STACKER_CACHE: Dict[str, Any] = {}

_FALLBACK_LOOP: asyncio.AbstractEventLoop | None = None
_FALLBACK_THREAD: threading.Thread | None = None
_FALLBACK_LOOP_LOCK = threading.Lock()
_FALLBACK_SHUTDOWN_REGISTERED = False

logger = logging.getLogger(__name__)

# ``_ROUTER`` is retained for compatibility with modules that import it.  The
# message bus supersedes the old ZeroMQ based router so it is simply ``None``
# here.
_ROUTER = None


def get_signal_backend(cfg: Dict[str, Any] | None = None) -> MessageBus:
    """Return a :class:`MessageBus` instance for publishing signals."""

    backend = (cfg or {}).get("signal_backend") if cfg else None
    return get_message_bus(backend)


# ---------------------------------------------------------------------------


def _wrap_ci(row: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap prediction rows with credible intervals if present."""

    if {"pred", "ci_lower", "ci_upper"}.issubset(row):
        row = row.copy()
        lower = row.pop("ci_lower")
        upper = row.pop("ci_upper")
        pred = row.pop("pred")
        row["prediction"] = {"mean": pred, "lower": lower, "upper": upper}
        y_true = row.get("y_true") or row.get("label")
        if y_true is not None:
            cov = evaluate_coverage([y_true], [lower], [upper])
            row["interval_covered"] = int(round(cov))
    elif {"pred", "pred_var"}.issubset(row):
        row = row.copy()
        row["prediction"] = {"mean": row.pop("pred"), "var": row.pop("pred_var")}
    return row


def _is_empty(df: pd.DataFrame) -> bool:
    return getattr(df, "empty", len(df) == 0)  # type: ignore[arg-type]


def _validate(df: pd.DataFrame) -> bool:
    try:
        return pipeline_anomaly.validate(df) if hasattr(df, "columns") else True
    except Exception:  # pragma: no cover - best effort
        logger.exception("Pipeline validation failed")
        return True


def _apply_residual(df: pd.DataFrame) -> pd.DataFrame:
    """Add residual stacker predictions to ``df`` when available."""

    if _residual_stacker is None:
        return df
    cols = getattr(df, "columns", None)
    if cols is None or "pred" not in cols or "features" not in cols:
        return df

    model_name = df.get("model_name")
    if model_name is None:
        name = "default"
    elif isinstance(model_name, pd.Series):
        name = model_name.iloc[0]
    else:
        name = str(model_name)

    model = _STACKER_CACHE.get(name)
    if model is None:
        model = _residual_stacker.load(name)
        _STACKER_CACHE[name] = model
    if model is None:
        return df

    feats = np.vstack(df["features"].to_numpy())
    base = df["pred"].to_numpy()
    residual = _residual_stacker.predict(feats, base, model)
    df = df.copy()
    df["pred"] = base + residual
    return df.drop(columns=["features"], errors="ignore")


def _prepare_rows(df: pd.DataFrame) -> list[Dict[str, Any]]:
    """Return processed rows ready for publishing.

    Shared between synchronous and asynchronous publishers.  The function
    performs residual stacking, pipeline validation and wraps credible
    intervals when present.  Any generated rows are recorded via
    :func:`event_store.event_writer.record`.
    """

    if _is_empty(df):
        return []
    df = _apply_residual(df)
    if not _validate(df):
        logger.warning("Pipeline anomaly detected; dropping batch")
        return []
    rows = [_wrap_ci(r) for r in df.to_dict(orient="records")]
    for row in rows:
        record_event("prediction", row)
    return rows


def _normalise_format(fmt: str) -> str:
    return (fmt or "json").strip().lower()


def _serialise_row(row: Dict[str, Any], fmt: str) -> Any:
    fmt = _normalise_format(fmt)
    if fmt == "json":
        return json.dumps(row).encode("utf-8")
    if fmt in {"dict", "python", "raw", "protobuf"}:
        return row
    raise ValueError(f"Unsupported signal format: {fmt}")


def _deserialise_message(msg: Any, fmt: str) -> Dict[str, Any] | Any | None:
    fmt = _normalise_format(fmt)
    if fmt == "json":
        if isinstance(msg, dict):
            return msg
        if isinstance(msg, (bytes, bytearray)):
            msg = msg.decode("utf-8", errors="ignore")
        if isinstance(msg, str):
            try:
                parsed = json.loads(msg)
            except json.JSONDecodeError:
                logger.exception("Failed to decode JSON signal message")
                return None
            if isinstance(parsed, dict):
                return parsed
            logger.warning("Decoded JSON signal is not a dict: %s", type(parsed))
            return None
        logger.warning("Received unsupported JSON payload type: %s", type(msg))
        return None
    if fmt in {"dict", "python", "raw", "protobuf"}:
        return msg
    raise ValueError(f"Unsupported signal format: {fmt}")


def _run_loop_forever(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    asyncio.set_event_loop(loop)
    ready.set()
    loop.run_forever()


def _ensure_shutdown_registered() -> None:
    global _FALLBACK_SHUTDOWN_REGISTERED
    if not _FALLBACK_SHUTDOWN_REGISTERED:
        atexit.register(_shutdown_fallback_loop)
        _FALLBACK_SHUTDOWN_REGISTERED = True


def _get_fallback_loop() -> asyncio.AbstractEventLoop:
    global _FALLBACK_LOOP, _FALLBACK_THREAD

    ready: threading.Event | None = None
    loop: asyncio.AbstractEventLoop | None
    thread: threading.Thread | None
    with _FALLBACK_LOOP_LOCK:
        loop = _FALLBACK_LOOP
        thread = _FALLBACK_THREAD

        if loop is not None:
            if loop.is_closed():
                loop = None
                thread = None
                _FALLBACK_LOOP = None
                _FALLBACK_THREAD = None
            elif thread is None or not thread.is_alive():
                try:
                    loop.close()
                except RuntimeError:
                    pass
                loop = None
                thread = None
                _FALLBACK_LOOP = None
                _FALLBACK_THREAD = None

        if loop is None:
            loop = asyncio.new_event_loop()
            ready = threading.Event()
            thread = threading.Thread(
                target=_run_loop_forever,
                args=(loop, ready),
                name="signal-queue-fallback-loop",
                daemon=True,
            )
            _FALLBACK_LOOP = loop
            _FALLBACK_THREAD = thread
            _ensure_shutdown_registered()
            thread.start()

    if ready is not None:
        ready.wait()

    if loop is None:
        raise RuntimeError("Failed to initialize fallback event loop")

    return loop


def _shutdown_fallback_loop() -> None:
    global _FALLBACK_LOOP, _FALLBACK_THREAD

    with _FALLBACK_LOOP_LOCK:
        loop = _FALLBACK_LOOP
        thread = _FALLBACK_THREAD
        _FALLBACK_LOOP = None
        _FALLBACK_THREAD = None

    if loop is None:
        return

    should_close = True

    if thread is not None and thread.is_alive():
        if not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        thread.join(timeout=5)
        if thread.is_alive():
            logger.warning("Fallback event loop thread did not terminate cleanly")
            should_close = False

    if should_close and not loop.is_closed():
        try:
            loop.close()
        except RuntimeError:
            logger.exception("Failed to close fallback event loop cleanly")


async def publish_rows(
    bus: MessageBus, rows: list[Dict[str, Any]], fmt: str
) -> None:
    """Publish ``rows`` to the signals topic."""

    for row in rows:
        payload = _serialise_row(row, fmt)
        await bus.publish(Topics.SIGNALS, payload)


def _log_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Signal publishing task failed")


def publish_dataframe(bus: MessageBus, df: pd.DataFrame, fmt: str = "json") -> None:
    """Synchronously publish each row of ``df`` to the signals topic."""

    rows = _prepare_rows(df)
    if not rows:
        return

    coro = publish_rows(bus, rows, fmt)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = _get_fallback_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        future.result()
    else:
        task = asyncio.create_task(coro)
        task.add_done_callback(_log_task_exception)


async def publish_dataframe_async(
    bus: MessageBus, df: pd.DataFrame, fmt: str = "json"
) -> None:
    """Asynchronously publish each row of ``df`` to the signals topic."""

    rows = _prepare_rows(df)
    if not rows:
        return
    await publish_rows(bus, rows, fmt)


async def iter_messages(
    bus: MessageBus, fmt: str = "json", sizer=None, meta_clf=None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield messages from the signals topic.

    Parameters
    ----------
    bus:
        Message bus to consume from.
    fmt:
        Serialization format used by the publisher.
    sizer:
        Optional position sizing callable.
    meta_clf:
        Optional meta-classifier used to filter predictions.  Messages with a
        ``prob`` field that the classifier deems unreliable (predicts ``0``)
        are dropped.

    When ``sizer`` is provided it is expected to expose the same interface as
    :meth:`risk_manager.RiskManager.adjust_size`.  Messages containing a
    ``prediction`` dictionary with ``mean`` and ``var`` fields will be
    transformed into a credible interval before being passed to ``sizer``.  The
    message's ``size`` field is replaced with the adjusted value.
    """

    async for raw_msg in bus.subscribe(Topics.SIGNALS):
        msg = _deserialise_message(raw_msg, fmt)
        if msg is None:
            continue
        if meta_clf is not None and isinstance(msg, dict):
            prob = msg.get("prob")
            if prob is not None:
                feat = pd.DataFrame(
                    {"prob": [prob], "confidence": [abs(float(prob) - 0.5) * 2]}
                )
                try:
                    if int(meta_clf.predict(feat)[0]) == 0:
                        continue
                except Exception:  # pragma: no cover - best effort
                    pass
        if sizer is not None and isinstance(msg, dict):
            pred = msg.get("prediction")
            symbol = msg.get("symbol")
            base_size = msg.get("size")
            ts = msg.get("Timestamp") or msg.get("timestamp")
            if (
                pred is not None
                and symbol is not None
                and base_size is not None
                and ts is not None
            ):
                ci: tuple[float, float] | None = None
                if "var" in pred and "mean" in pred:
                    std = float(np.sqrt(max(pred["var"], 0.0)))
                    mean = float(pred["mean"])
                    ci = (mean - 1.96 * std, mean + 1.96 * std)
                elif {"lower", "upper"}.issubset(pred):
                    ci = (float(pred["lower"]), float(pred["upper"]))
                direction = 1 if float(base_size) >= 0 else -1
                try:
                    sized = sizer(
                        symbol,
                        abs(float(base_size)),
                        ts,
                        direction,
                        cred_interval=ci,
                    )
                    msg["size"] = sized * direction
                except Exception:  # pragma: no cover - sizing is best effort
                    pass
        yield msg
