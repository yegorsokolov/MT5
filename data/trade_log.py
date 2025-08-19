"""Simple trade log using SQLite for orders, fills and positions."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List
from datetime import datetime


class TradeLog:
    """Persist executed orders, fills and open positions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path.as_posix())
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                volume REAL,
                price REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                volume REAL,
                price REAL,
                FOREIGN KEY(order_id) REFERENCES orders(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                volume REAL,
                avg_price REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS thresholds (
                order_id INTEGER PRIMARY KEY,
                base_tp REAL,
                base_sl REAL,
                adaptive_tp REAL,
                adaptive_sl REAL,
                FOREIGN KEY(order_id) REFERENCES orders(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS survival (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER,
                timestamp TEXT,
                probability REAL,
                FOREIGN KEY(order_id) REFERENCES orders(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS confirmations (
                order_id INTEGER PRIMARY KEY,
                score REAL,
                FOREIGN KEY(order_id) REFERENCES orders(id)
            )
            """,
        )
        self.conn.commit()

    @staticmethod
    def _dt(val: Any) -> str:
        if isinstance(val, datetime):
            return val.isoformat()
        return str(val)

    def record_order(self, order: Dict[str, Any]) -> int:
        """Insert a new order and return its id."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO orders(timestamp, symbol, side, volume, price) VALUES(?,?,?,?,?)",
            (
                self._dt(order.get("timestamp")),
                order.get("symbol"),
                order.get("side"),
                float(order.get("volume", 0)),
                float(order.get("price", 0)),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def record_confirmation(self, order_id: int, score: float) -> None:
        """Persist a confirmation score for ``order_id``."""

        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO confirmations(order_id, score) VALUES(?,?)",
            (order_id, float(score)),
        )
        self.conn.commit()

    def record_fill(self, fill: Dict[str, Any]) -> None:
        """Record a fill and update positions."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO fills(order_id, timestamp, symbol, side, volume, price) VALUES(?,?,?,?,?,?)",
            (
                fill.get("order_id"),
                self._dt(fill.get("timestamp")),
                fill.get("symbol"),
                fill.get("side"),
                float(fill.get("volume", 0)),
                float(fill.get("price", 0)),
            ),
        )

        sym = fill.get("symbol")
        side = (fill.get("side") or "").upper()
        vol = float(fill.get("volume", 0))
        if side == "SELL":
            vol = -vol
        price = float(fill.get("price", 0))
        row = cur.execute(
            "SELECT volume, avg_price FROM positions WHERE symbol=?", (sym,)
        ).fetchone()
        if row:
            cur_vol, cur_price = row
            new_vol = cur_vol + vol
            if abs(new_vol) < 1e-9:
                cur.execute("DELETE FROM positions WHERE symbol=?", (sym,))
            else:
                new_price = (cur_price * cur_vol + price * vol) / new_vol
                cur.execute(
                    "UPDATE positions SET volume=?, avg_price=? WHERE symbol=?",
                    (new_vol, new_price, sym),
                )
        else:
            cur.execute(
                "INSERT INTO positions(symbol, volume, avg_price) VALUES(?,?,?)",
                (sym, vol, price),
            )
        self.conn.commit()

    def get_open_positions(self) -> List[Dict[str, float]]:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT symbol, volume, avg_price FROM positions"
        ).fetchall()
        return [
            {"symbol": r[0], "volume": r[1], "avg_price": r[2]} for r in rows
        ]

    def sync_mt5_positions(self, mt5_positions: Iterable[Any]) -> None:
        """Replace stored positions with the given MT5 positions."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM positions")
        for pos in mt5_positions or []:
            cur.execute(
                "INSERT INTO positions(symbol, volume, avg_price) VALUES(?,?,?)",
                (getattr(pos, "symbol", None), getattr(pos, "volume", 0), getattr(pos, "price_open", 0)),
            )
        self.conn.commit()

    def record_thresholds(
        self,
        order_id: int,
        base_tp: float,
        base_sl: float,
        adaptive_tp: float,
        adaptive_sl: float,
    ) -> None:
        """Persist default and adaptive exit levels for ``order_id``."""
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO thresholds(order_id, base_tp, base_sl, adaptive_tp, adaptive_sl)
            VALUES(?,?,?,?,?)
            """,
            (order_id, base_tp, base_sl, adaptive_tp, adaptive_sl),
        )
        self.conn.commit()

    def get_thresholds(self, order_id: int) -> Dict[str, float] | None:
        """Return stored thresholds for ``order_id`` if available."""
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT base_tp, base_sl, adaptive_tp, adaptive_sl FROM thresholds WHERE order_id=?",
            (order_id,),
        ).fetchone()
        if row:
            return {
                "base_tp": row[0],
                "base_sl": row[1],
                "adaptive_tp": row[2],
                "adaptive_sl": row[3],
            }
        return None

    def record_survival(
        self,
        order_id: int,
        probability: float,
        timestamp: datetime | None = None,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO survival(order_id, timestamp, probability)
            VALUES(?,?,?)
            """,
            (order_id, self._dt(timestamp or datetime.utcnow()), float(probability)),
        )
        self.conn.commit()

    def get_survival(self, order_id: int) -> List[float]:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT probability FROM survival WHERE order_id=? ORDER BY timestamp",
            (order_id,),
        ).fetchall()
        return [r[0] for r in rows]
