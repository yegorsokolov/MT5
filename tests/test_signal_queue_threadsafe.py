import json
import threading

import pandas as pd
import pytest

import signal_queue


class _ThreadSafeBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.published: list[tuple[str, object]] = []

    async def publish(self, topic: str, message: object) -> None:
        with self._lock:
            self.published.append((topic, message))


@pytest.fixture(autouse=True)
def _cleanup_fallback_loop():
    signal_queue._shutdown_fallback_loop()
    yield
    signal_queue._shutdown_fallback_loop()


def _validate_stub(_: object) -> bool:
    return True


def _record_stub(*_: object, **__: object) -> None:
    return None


def test_publish_dataframe_threadsafe(monkeypatch):
    bus = _ThreadSafeBus()
    df_one = pd.DataFrame({"Timestamp": ["2028", "2029"], "prob": [0.21, 0.34]})
    df_two = pd.DataFrame({"Timestamp": ["2030", "2031"], "prob": [0.55, 0.89]})

    monkeypatch.setattr(signal_queue, "record_event", _record_stub)
    monkeypatch.setattr(signal_queue.pipeline_anomaly, "validate", _validate_stub)

    barrier = threading.Barrier(3)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(frame: pd.DataFrame) -> None:
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            return
        try:
            signal_queue.publish_dataframe(bus, frame)
        except BaseException as exc:  # noqa: BLE001 - capture unexpected failures
            with errors_lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(df_one,), name="publisher-one"),
        threading.Thread(target=worker, args=(df_two,), name="publisher-two"),
    ]

    for thread in threads:
        thread.start()

    try:
        barrier.wait(timeout=5)
    except threading.BrokenBarrierError:
        pass

    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads), "Worker threads did not finish"
    assert not errors, f"Unexpected errors raised: {errors!r}"

    expected_probs = sorted(
        list(df_one["prob"].astype(float)) + list(df_two["prob"].astype(float))
    )
    published_probs = sorted(
        json.loads(payload.decode("utf-8"))["prob"] for _, payload in bus.published
    )

    assert len(published_probs) == len(expected_probs)
    assert published_probs == expected_probs
