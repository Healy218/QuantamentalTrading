import backtrader as bt
import backtrader.indicators as btind

class ReversalAtKeyLevelsStrategy(bt.Strategy):
    def __init__(self):
        self.vwap = btind.VWAP(self.data)  # Backtrader's VWAP uses high, low, close, volume
        self.order = None
        self.trade_state = 0  # 0: no position, 1: long (call), -1: short (put)

    def next(self):
        if self.order:
            return

        # Calculate candlestick properties
        body = abs(self.data.close[0] - self.data.open[0])
        lower_wick = min(self.data.close[0], self.data.open[0]) - self.data.low[0]
        is_hammer = (lower_wick > 2 * body) and (self.data.close[0] > self.vwap[0])
        is_doji = (body < 0.0001 * self.data.close[0]) and (self.data.close[0] < self.vwap[0])

        if not self.position:
            # Buy Call: Hammer at VWAP support
            if is_hammer:
                self.order = self.buy()
                self.trade_state = 1
                # print(f"Buy CALL order opened on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

            # Buy Put: Doji at VWAP resistance
            elif is_doji:
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