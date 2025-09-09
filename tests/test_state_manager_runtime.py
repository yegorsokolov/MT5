import os
import sys
import importlib
from pathlib import Path

import joblib
import pytest


def reload_state_manager(tmp_path):
    os.environ['MT5BOT_STATE_DIR'] = str(tmp_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import state_manager
    importlib.reload(state_manager)
    return state_manager


def test_state_dir_configurable_and_permissions(tmp_path):
    sm = reload_state_manager(tmp_path)
    sm.save_runtime_state('ts', [], [], account_id='123')
    assert sm._STATE_DIR == Path(tmp_path)
    assert (tmp_path.stat().st_mode & 0o777) == 0o700
    assert sm._runtime_state_file('123').exists()


def test_load_runtime_state_no_fallback(tmp_path):
    sm = reload_state_manager(tmp_path)
    sm._ensure_state_dir()
    joblib.dump({'legacy': True}, sm._STATE_FILE)
    assert sm.legacy_runtime_state_exists()
    assert sm.load_runtime_state(account_id='123') is None


def test_migrate_runtime_state(tmp_path):
    sm = reload_state_manager(tmp_path)
    sm._ensure_state_dir()
    data = {'last_timestamp': 't'}
    joblib.dump(data, sm._STATE_FILE)
    new_path = sm.migrate_runtime_state('123')
    assert new_path == sm._runtime_state_file('123')
    assert new_path.exists()
    assert not sm._STATE_FILE.exists()
    assert joblib.load(new_path) == data


def test_runtime_state_file_invalid_account(tmp_path):
    sm = reload_state_manager(tmp_path)
    with pytest.raises(ValueError):
        sm._runtime_state_file('../bad')
