import os
import sys
import importlib
import types
import pickle
from pathlib import Path

joblib_stub = types.ModuleType('joblib')


def _joblib_dump(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)
    return [str(path)]


def _joblib_load(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


joblib_stub.dump = _joblib_dump
joblib_stub.load = _joblib_load
joblib_stub.dumps = pickle.dumps
joblib_stub.loads = pickle.loads

sys.modules.setdefault('joblib', joblib_stub)

import joblib
import pytest


def reload_state_manager(tmp_path):
    os.environ['MT5BOT_STATE_DIR'] = str(tmp_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    dummy_module = types.ModuleType('config_models')

    class DummyConfig:
        def update_from(self, other):
            pass

    class DummyError(Exception):
        pass

    dummy_module.AppConfig = DummyConfig
    dummy_module.ConfigError = DummyError
    sys.modules['config_models'] = dummy_module

    utils_module = types.ModuleType('utils')

    def load_config_stub(path='config.yaml'):
        return DummyConfig()

    utils_module.load_config = load_config_stub
    sys.modules['utils'] = utils_module

    crypto_module = types.ModuleType('crypto_utils')
    crypto_module._load_key = lambda name: b''
    crypto_module.encrypt = lambda data, key: data
    crypto_module.decrypt = lambda data, key: data
    sys.modules['crypto_utils'] = crypto_module

    sys.modules['joblib'] = joblib_stub

    sys.modules.pop('state_manager', None)
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
    sm.save_runtime_state('legacy', [], [], account_id=None)
    assert sm.legacy_runtime_state_exists()
    assert sm.load_runtime_state(account_id='123') is None


def test_migrate_runtime_state(tmp_path):
    sm = reload_state_manager(tmp_path)
    sm._ensure_state_dir()
    data = {'last_timestamp': 't', 'open_positions': [], 'model_versions': []}
    sm.save_runtime_state('t', [], [], account_id=None)
    new_path = sm.migrate_runtime_state('123')
    assert new_path == sm._runtime_state_file('123')
    assert new_path.exists()
    assert not sm._STATE_FILE.exists()
    with open(new_path, 'rb') as fh:
        assert pickle.load(fh) == data


def test_migrate_runtime_state_merges_existing(tmp_path):
    sm = reload_state_manager(tmp_path)
    sm._ensure_state_dir()
    legacy = {
        'last_timestamp': '2024-01-01T00:00:00+00:00',
        'open_positions': [{'ticket': 1}, {'ticket': 2}],
        'model_versions': ['legacy', 'shared'],
        'model_weights': {'legacy': 0.1, 'shared': 0.5},
        'feature_scalers': {'scaler': 'legacy'},
    }
    sm.save_runtime_state(
        legacy['last_timestamp'],
        legacy['open_positions'],
        legacy['model_versions'],
        model_weights=legacy['model_weights'],
        feature_scalers=legacy['feature_scalers'],
        account_id=None,
    )
    new_state = {
        'last_timestamp': '2023-12-31T00:00:00+00:00',
        'open_positions': [{'ticket': 2}, {'ticket': 3}],
        'model_versions': ['current'],
        'model_weights': {'current': 0.2, 'shared': 0.9},
    }
    new_path = sm._runtime_state_file('123')
    sm.save_runtime_state(
        new_state['last_timestamp'],
        new_state['open_positions'],
        new_state['model_versions'],
        model_weights=new_state['model_weights'],
        account_id='123',
    )

    merged_path = sm.migrate_runtime_state('123')

    assert merged_path == new_path
    assert not sm._STATE_FILE.exists()
    with open(merged_path, 'rb') as fh:
        merged = pickle.load(fh)

    # The most recent timestamp (legacy) is retained
    assert merged['last_timestamp'] == legacy['last_timestamp']
    # New positions stay first while legacy-only entries are appended
    assert merged['open_positions'][:2] == new_state['open_positions']
    assert merged['open_positions'][2:] == [legacy['open_positions'][0]]
    # Versions are deduplicated while preserving order preference for new data
    assert merged['model_versions'] == ['current', 'legacy', 'shared']
    # Weights are merged with new values taking precedence
    assert merged['model_weights'] == {
        'legacy': 0.1,
        'shared': 0.9,
        'current': 0.2,
    }
    # Feature scalers from the legacy state are kept when absent from the new state
    assert merged['feature_scalers'] == legacy['feature_scalers']


def test_migrate_runtime_state_existing_without_legacy(tmp_path):
    sm = reload_state_manager(tmp_path)
    sm._ensure_state_dir()
    new_path = sm._runtime_state_file('123')
    sm.save_runtime_state('existing', [], [], account_id='123')

    assert sm.migrate_runtime_state('123') == new_path


def test_migrate_runtime_state_missing_source(tmp_path):
    sm = reload_state_manager(tmp_path)
    with pytest.raises(FileNotFoundError):
        sm.migrate_runtime_state('123')


def test_runtime_state_file_invalid_account(tmp_path):
    sm = reload_state_manager(tmp_path)
    with pytest.raises(ValueError):
        sm._runtime_state_file('../bad')
