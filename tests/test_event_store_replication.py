from pathlib import Path
import types
import logging
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from event_store import EventStore


def test_event_store_replication(monkeypatch, tmp_path, caplog):
    calls = []

    def fake_sync(db, ds, backend=None):
        calls.append((db, ds, backend))
        return True

    monkeypatch.setattr('event_store.state_sync', types.SimpleNamespace(sync_event_store=fake_sync))
    store = EventStore(tmp_path / 'events.db', backend='nfs://remote')
    caplog.set_level(logging.INFO)
    store.record('feature', {'x': 1})
    assert calls, 'sync_event_store not called'
    assert 'Replicated event store' in caplog.text
