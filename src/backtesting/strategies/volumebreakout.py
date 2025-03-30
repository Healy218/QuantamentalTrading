import backtrader as bt
import backtrader.indicators as btind

class VolumeBreakoutStrategy(bt.Strategy):
    params = (
        ('lookback', 5),
        ('volume_multiplier', 3),
    )

    def __init__(self):
        self.avg_volume = btind.SMA(self.data.volume, period=20)
        self.high_break = btind.Highest(self.data.high, period=self.p.lookback)
        self.low_break = btind.Lowest(self.data.low, period=self.p.lookback)
        self.order = None
        self.trade_state = 0  # 0: no position, 1: long (call), -1: short (put)

    def next(self):
        if self.order:
            return

        if not self.position:
            # Buy Call: Break above high with volume spike
            if (self.data.close[0] > self.high_break[-1]) and \
               (self.data.volume[0] > self.avg_volume[0] * self.p.volume_multiplier):
                self.order = self.buy()
                self.trade_state = 1
                # print(f"Buy CALL order opened on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

            # Buy Put: Break below low with volume spike
            elif (self.data.close[0] < self.low_break[-1]) and \
                 (self.data.volume[0] > self.avg_volume[0] * self.p.volume_multiplier):
                self.order = self.sell()
                self.trade_state = -1
                # print(f"Buy PUT order opened on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

        elif self.trade_state == 1:  # Long position
            self.order = self.close()
            self.trade_state = 0
            # print(f"CALL position closed on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

        elif self.trade_state == -1:  # Short position
            self.order = self.close()
            self.trade_state = 0
            # print(f"PUT position closed on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                # print(f"BUY EXECUTED, Price: {order.executed.price:.2f}")
                pass
            elif order.issell():
                # print(f"SELL EXECUTED, Price: {order.executed.price:.2f}")
                pass
        self.order = None