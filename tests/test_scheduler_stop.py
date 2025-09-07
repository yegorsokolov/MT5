import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import scheduler


def test_stop_scheduler_cancels_tasks_and_stops_loop():
    scheduler.stop_scheduler()
    scheduler._schedule_jobs([("dummy", lambda: None)])
    assert scheduler._thread is not None and scheduler._thread.is_alive()
    tasks = list(scheduler._tasks)
    scheduler.stop_scheduler()
    assert scheduler._loop is None
    assert scheduler._thread is None or not scheduler._thread.is_alive()
    assert scheduler._tasks == []
    assert all(t.cancelled() for t in tasks)
