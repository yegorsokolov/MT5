import asyncio
import gc
import sqlite3
import sys
import pathlib

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.config_service import SQLiteConfigStore


def test_store_version_and_watch(tmp_path):
    async def scenario():
        db_path = tmp_path / "cfg.db"
        store = SQLiteConfigStore(str(db_path))

        q = store.register()
        first = await store.set("alpha", "1", "bob", "init")
        assert first["version"] == 1
        await q.get()  # consume first event

        waiter = asyncio.create_task(q.get())
        second = await store.set("alpha", "2", "bob", "bump")
        event = await asyncio.wait_for(waiter, timeout=1)
        assert event["version"] == 2
        assert event["value"] == "2"
        assert second["version"] == 2

        count = await asyncio.to_thread(
            lambda: store.conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0]
        )
        assert count == 2

    asyncio.run(scenario())


def test_close_closes_connection(tmp_path):
    db_path = tmp_path / "close.db"
    store = SQLiteConfigStore(str(db_path))
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.conn.execute("SELECT 1")


def test_context_manager_closes_connection(tmp_path):
    db_path = tmp_path / "ctx.db"
    with SQLiteConfigStore(str(db_path)) as store:
        store.conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        store.conn.execute("SELECT 1")


def test_del_closes_connection(tmp_path):
    db_path = tmp_path / "del.db"
    store = SQLiteConfigStore(str(db_path))
    conn = store.conn
    del store
    gc.collect()
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
