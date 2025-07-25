import backtrader as bt

class DonchianATRStrategy(bt.Strategy):
    """Donchian channel breakout with ATR based stop."""

    params = dict(donchian=20, atr_period=14, atr_mult=2)

    def __init__(self):
        self.don_high = bt.ind.Highest(self.data.high, period=self.p.donchian)
        self.don_low = bt.ind.Lowest(self.data.low, period=self.p.donchian)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.stop_price = None

    def next(self):
        if not self.position:
            if self.data.close[0] > self.don_high[-1]:
                self.buy()
                self.stop_price = self.data.close[0] - self.atr[0] * self.p.atr_mult
        else:
            if self.data.close[0] < self.stop_price:
                self.close()
            else:
                self.stop_price = max(self.stop_price, self.data.close[0] - self.atr[0] * self.p.atr_mult)
