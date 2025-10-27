# scripts/check_mt5_direct.py
import os, sys

# Prefer your mt5 shim if present; else try MetaTrader5 import
try:
    import mt5 as mt5
except Exception:
    import MetaTrader5 as mt5  # type: ignore

missing = [k for k in ("MT5_LOGIN","MT5_PASSWORD","MT5_SERVER") if not os.environ.get(k)]
if missing:
    print("Missing env:", ", ".join(missing))
    sys.exit(2)

login = int(os.environ["MT5_LOGIN"])
password = os.environ["MT5_PASSWORD"]
server = os.environ["MT5_SERVER"]

ok = mt5.initialize()
print("initialize:", ok)
if not ok:
    try: print("last_error:", mt5.last_error())
    except Exception: pass
    sys.exit(3)

ok = mt5.login(login, password=password, server=server)
print("login:", ok)
if not ok:
    try: print("last_error:", mt5.last_error())
    except Exception: pass
    sys.exit(4)

ti = mt5.terminal_info()
print("terminal_info:",
      getattr(ti, "build", None),
      getattr(ti, "connected", None),
      getattr(ti, "path", None))
print("OK")
