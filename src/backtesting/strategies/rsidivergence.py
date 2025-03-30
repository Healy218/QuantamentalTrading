import backtrader as bt
import backtrader.indicators as btind

class RSIDivergenceStrategy(bt.Strategy):
    params = (
        ('rsi_period', 5),
    )

    def __init__(self):
        self.rsi = btind.RSI(self.data.close, period=self.p.rsi_period)
        self.order = None
        self.trade_state = 0  # 0: no position, 1: long (call), -1: short (put)

    def next(self):
        if self.order or len(self.data) < 2:  # Need at least 2 bars for diff
            return

        price_diff = self.data.close[0] - self.data.close[-1]
        rsi_diff = self.rsi[0] - self.rsi[-1]

        bullish_div = (price_diff > 0) and (rsi_diff < 0) and (self.rsi[0] < 30)
        bearish_div = (price_diff < 0) and (rsi_diff > 0) and (self.rsi[0] > 70)

        if not self.position:
            # Buy Call: Bullish divergence
            if bullish_div:
                self.order = self.buy()
                self.trade_state = 1
                # print(f"Buy CALL order opened on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

            # Buy Put: Bearish divergence
            elif bearish_div:
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