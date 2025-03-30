import backtrader as bt
from backtrader import indicators as btind

class MomentumBurstStrategy(bt.Strategy):
    params = (
        ('ema_period', 5),
        ('price_threshold', 0.001),
        ('volume_multiplier', 2),
    )

    def __init__(self):
        self.ema = btind.EMA(self.data.close, period=self.p.ema_period)
        self.avg_volume = btind.SMA(self.data.volume, period=20)
        self.price_change = btind.PercentChange(self.data.close, period=1)
        self.order = None
        self.trade_state = 0 # 0: no position, 1: long (call), -1: short (put)

    def next(self):
        if self.order:
            return

        if not self.position:
            if (self.data.close[0] > self.ema[0]) and \
               (self.price_change[0] > self.p.price_threshold) and \
               (self.data.volume[0] > self.avg_volume[0] * self.p.volume_multiplier):
                self.order = self.buy()
                self.trade_state = 1
                print(f"Buy CALL order opened on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

            elif (self.data.close[0] < self.ema[0]) and \
                 (self.price_change[0] < -self.p.price_threshold) and \
                 (self.data.volume[0] > self.avg_volume[0] * self.p.volume_multiplier):
                self.order = self.sell() # Assuming 'sell' opens a short position for a put
                self.trade_state = -1
                print(f"Buy PUT order opened on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

        elif self.trade_state == 1: # If in a long (call) position, for simplicity, let's close after one period
            self.order = self.close()
            self.trade_state = 0
            print(f"CALL position closed on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

        elif self.trade_state == -1: # If in a short (put) position, close after one period
            self.order = self.close()
            self.trade_state = 0
            print(f"PUT position closed on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
            elif order.issell():
                print(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"Order Canceled/Margin/Rejected: {order.status}")
        self.order = None