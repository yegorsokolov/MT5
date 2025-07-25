from freqtrade.strategy import IStrategy
import pandas as pd
import talib.abstract as ta

class BasicTemplate(IStrategy):
    """Simple moving average crossover strategy for Freqtrade."""

    timeframe = '5m'
    minimal_roi = {"0": 0.02}
    stoploss = -0.1
    trailing_stop = True

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['ma_fast'] = ta.SMA(dataframe['close'], timeperiod=10)
        dataframe['ma_slow'] = ta.SMA(dataframe['close'], timeperiod=30)
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        cond = (dataframe['ma_fast'] > dataframe['ma_slow']) & (dataframe['rsi'] > 55)
        dataframe.loc[cond, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = ta.CROSSOVER(dataframe['ma_slow'], dataframe['ma_fast'])
        return dataframe
