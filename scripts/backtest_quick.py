#!/usr/bin/env python
from __future__ import annotations
import os, sys, math, time, datetime as dt
from dataclasses import dataclass

# Prefer your shim; fall back to MetaTrader5 if available
try:
    import mt5 as mt5
except Exception:
    import MetaTrader5 as mt5  # type: ignore

import pandas as pd

@dataclass
class Config:
    symbol: str = os.environ.get("BT_SYMBOL", "EURUSD")
    timeframe: str = os.environ.get("BT_TIMEFRAME", "M5")   # M1,M5,M15,M30,H1,H4,D1
    start: str = os.environ.get("BT_START", "2025-09-01")
    end: str   = os.environ.get("BT_END",   "2025-10-01")
    fast: int  = int(os.environ.get("BT_FAST", "12"))
    slow: int  = int(os.environ.get("BT_SLOW", "26"))
    sl_points: int = int(os.environ.get("BT_SL", "250"))     # backtest-only
    tp_points: int = int(os.environ.get("BT_TP", "500"))

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

def ensure_init_and_login():
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")
    if not all([login, password, server]):
        sys.exit("Missing MT5_LOGIN/MT5_PASSWORD/MT5_SERVER")
    if not mt5.initialize():
        sys.exit(f"initialize() failed: {getattr(mt5,'last_error',lambda:('?','?'))()}")
    if not mt5.login(int(login), password=password, server=server):
        sys.exit(f"login() failed: {getattr(mt5,'last_error',lambda:('?','?'))()}")

def fetch_df(cfg: Config) -> pd.DataFrame:
    tf = TF_MAP[cfg.timeframe]
    ts_start = int(pd.Timestamp(cfg.start, tz='UTC').timestamp())
    ts_end   = int(pd.Timestamp(cfg.end, tz='UTC').timestamp())
    rates = mt5.copy_rates_range(cfg.symbol, tf, ts_start, ts_end)
    if rates is None or len(rates) == 0:
        sys.exit("No data returned. Check symbol/timeframe/dates.")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()

def run():
    cfg = Config()
    ensure_init_and_login()
    df = fetch_df(cfg)
    df['fast'] = ema(df['close'], cfg.fast)
    df['slow'] = ema(df['close'], cfg.slow)
    df['signal'] = 0
    df.loc[df['fast'] > df['slow'], 'signal'] = 1
    df.loc[df['fast'] < df['slow'], 'signal'] = -1
    df['trade'] = df['signal'].diff().fillna(0)

    # Very rough PnL: enter at next bar open, exit on opposite signal or SL/TP in points
    entry_price = None
    direction = 0
    pnl_points = 0
    winners = losers = 0
    entries = 0

    # pip value approximator for 5-digit majors: 1 point = 0.00001
    def price_to_points(p0, p1):
        return int(round((p1 - p0) / 0.00001))

    for t, row in df.iterrows():
        if row['trade'] != 0:
            # close previous
            if direction != 0 and entry_price is not None:
                exit_price = row['open']
                move = price_to_points(entry_price, exit_price)
                if direction < 0:
                    move = -move
                pnl_points += move
                if move >= 0: winners += 1
                else: losers += 1

            # open new
            direction = int(row['signal'])
            entry_price = row['open']
            entries += 1
            continue

        # sl/tp checks using bar extremes
        if direction != 0 and entry_price is not None:
            if direction > 0:
                # long
                if price_to_points(entry_price, row['low']) <= -cfg.sl_points:
                    pnl_points += -cfg.sl_points
                    losers += 1
                    entry_price = None; direction = 0
                elif price_to_points(entry_price, row['high']) >= cfg.tp_points:
                    pnl_points += cfg.tp_points
                    winners += 1
                    entry_price = None; direction = 0
            else:
                # short
                if price_to_points(entry_price, row['high']) >= cfg.sl_points:
                    pnl_points += -cfg.sl_points
                    losers += 1
                    entry_price = None; direction = 0
                elif price_to_points(entry_price, row['low']) <= -cfg.tp_points:
                    pnl_points += cfg.tp_points
                    winners += 1
                    entry_price = None; direction = 0

    # Close any open at last close
    if direction != 0 and entry_price is not None:
        last_close = df.iloc[-1]['close']
        move = price_to_points(entry_price, last_close)
        if direction < 0:
            move = -move
        pnl_points += move
        if move >= 0: winners += 1
        else: losers += 1

    print(f"[BACKTEST] {cfg.symbol} {cfg.timeframe} {cfg.start}→{cfg.end}")
    print(f" entries={entries} pnl_points={pnl_points} winners={winners} losers={losers}")

if __name__ == "__main__":
    run()
