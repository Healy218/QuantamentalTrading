import backtrader as bt
import backtrader.indicators as btind

class MACDZeroCrossStrategy(bt.Strategy):
    params = (
        ('fast', 3),
        ('slow', 10),
        ('signal', 5),
        ('ema_period', 8),
    )

    def __init__(self):
        self.macd = btind.MACD(self.data.close, period_me1=self.p.fast, 
                              period_me2=self.p.slow, period_signal=self.p.signal)
        self.ema = btind.EMA(self.data.close, period=self.p.ema_period)
        self.order = None
        self.trade_state = 0  # 0: no position, 1: long (call), -1: short (put)

    def next(self):
        if self.order:
            return

        if not self.position:
            # Buy Call: MACD crosses above 0 and price above EMA
            if (self.macd.macd[0] > 0 and self.macd.macd[-1] < 0) and \
               (self.data.close[0] > self.ema[0]):
                self.order = self.buy()
                self.trade_state = 1
                # print(f"Buy CALL order opened on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

            # Buy Put: MACD crosses below 0 and price below EMA
            elif (self.macd.macd[0] < 0 and self.macd.macd[-1] > 0) and \
                 (self.data.close[0] < self.ema[0]):
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