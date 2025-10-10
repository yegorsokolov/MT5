import importlib
import sys
import types
from pathlib import Path

import pytest

pandas_mod = pytest.importorskip("pandas")


@pytest.fixture
def load_mt5_direct(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    def _loader(stub):
        sys.modules.pop("brokers.mt5_direct", None)
        sys.modules.pop("MetaTrader5", None)
        # Provide a lightweight placeholder so the import succeeds even when the
        # real dependency is unavailable in the test environment.
        monkeypatch.setitem(sys.modules, "MetaTrader5", types.SimpleNamespace())
        module = importlib.import_module("brokers.mt5_direct")
        monkeypatch.setattr(module, "_mt5", stub, raising=False)

        # Ensure tests always exercise the pandas code path regardless of the
        # configured dataframe backend.
        monkeypatch.setattr(module, "pd", pandas_mod, raising=False)
        monkeypatch.setattr(module, "_pd", pandas_mod, raising=False)
        monkeypatch.setattr(module, "IS_CUDF", False, raising=False)

        success_codes = {
            getattr(stub, name)
            for name in getattr(module, "_SUCCESS_RETCODE_NAMES", tuple())
            if hasattr(stub, name)
        }
        if not success_codes:
            success_codes = {0}
        monkeypatch.setattr(module, "_SUCCESS_RETCODES", success_codes, raising=False)

        for attr in ("ORDER_TYPE_BUY", "ORDER_TYPE_SELL", "TRADE_ACTION_DEAL"):
            if hasattr(stub, attr):
                monkeypatch.setattr(module, attr, getattr(stub, attr), raising=False)
        return module

    return _loader


def test_order_send_success(load_mt5_direct):
    class Stub:
        TRADE_RETCODE_DONE = 10009

        def order_send(self, request):
            return types.SimpleNamespace(retcode=10009, price=request.get("price"))

    module = load_mt5_direct(Stub())
    request = {"price": 1.2345, "type": module.ORDER_TYPE_BUY}
    result = module.order_send(request)
    assert result.retcode == 10009


def test_order_send_raises_on_failure(load_mt5_direct):
    class Stub:
        TRADE_RETCODE_DONE = 10009

        def order_send(self, request):
            return types.SimpleNamespace(
                retcode=5,
                comment="volume too high",
                request_id=42,
            )

        def last_error(self):
            return (5001, "volume not allowed")

    module = load_mt5_direct(Stub())
    request = {"price": 1.25, "type": module.ORDER_TYPE_BUY}

    with pytest.raises(module.MT5OrderError) as excinfo:
        module.order_send(request)

    err = excinfo.value
    assert err.code == 5
    assert "volume too high" in str(err)
    assert err.details["last_error_code"] == 5001
    assert err.details["request"] == request
    assert "resolution" in err.details
    assert "volume" in " ".join(err.details["resolution"]["steps"])


def test_symbol_select_failure_raises(load_mt5_direct):
    class Stub:
        def symbol_select(self, symbol, enable):
            return False

        def last_error(self):
            return (4301, "unknown symbol")

    module = load_mt5_direct(Stub())

    with pytest.raises(module.MT5Error) as excinfo:
        module.symbol_select("EURUSD")

    err = excinfo.value
    assert err.code == 4301
    assert "EURUSD" in str(err)
    assert "resolution" in err.details


def test_copy_ticks_from_none_raises(load_mt5_direct):
    class Stub:
        def copy_ticks_from(self, symbol, from_time, count, flags):
            return None

        def last_error(self):
            return (4401, "no data")

    module = load_mt5_direct(Stub())

    with pytest.raises(module.MT5Error) as excinfo:
        module.copy_ticks_from("EURUSD", 0, 10, 0)

    err = excinfo.value
    assert err.code == 4401
    assert "no data" in str(err)
    assert "resolution" in err.details


def test_find_mt5_symbol_discovers_suffix(load_mt5_direct):
    class Stub:
        def symbol_info(self, symbol):
            return None

        def symbols_total(self):
            return 2

        def symbols_get(self, *args):
            if args == (0, 2):
                raise TypeError("slice not supported")
            return [types.SimpleNamespace(name="EURUSD.m"), types.SimpleNamespace(name="US500")]  # noqa: PIE800

    module = load_mt5_direct(Stub())

    resolved = module._find_mt5_symbol("EURUSD")
    assert resolved == "EURUSD.m"


def test_find_mt5_symbol_handles_iterables(load_mt5_direct):
    class Stub:
        def symbol_info(self, symbol):
            return None

        def symbols_total(self):
            return 0

        def symbols_get(self, *args):
            if not args:
                return [("XAUUSD", "Gold"), {"symbol": "US500"}]
            raise TypeError("unexpected args")

    module = load_mt5_direct(Stub())

    assert module._find_mt5_symbol("XAUUSD") == "XAUUSD"
    assert module._find_mt5_symbol("US500") == "US500"
