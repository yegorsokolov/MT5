"""Allow invoking the toolkit as ``python -m mt5_toolkit``.

This is a lightweight compatibility shim that simply forwards to the
:mod:`mt5` package dispatcher.  It is useful in environments where the name
``mt5`` collides with other packages or executables installed globally.
"""

from mt5.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
