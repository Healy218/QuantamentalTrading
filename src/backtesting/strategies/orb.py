import backtrader as bt
import numpy as np

class OpeningRangeBreakoutStrategy(bt.Strategy):
    params = (
        ('orb_period', 15),  # Number of periods for opening range
    )

    def __init__(self):
        self.order = None
        self.trade_state = 0  # 0: no position, 1: long (call), -1: short (put)
        self.data_ready = False
        self.orb_high = None
        self.orb_low = None
        self.bar_count = 0

    def next(self):
        if self.order:
            return

        self.bar_count += 1

        # Calculate ORB in the first orb_period bars
        if self.bar_count <= self.p.orb_period:
            if self.bar_count == self.p.orb_period:
                self.orb_high = max(self.data.high.get(size=self.p.orb_period))
                self.orb_low = min(self.data.low.get(size=self.p.orb_period))
                self.data_ready = True
            return

        # Only proceed after ORB is calculated
        if not self.data_ready:
            return

        # Get current and previous volume
        current_volume = self.data.volume[0]
        prev_volume = self.data.volume[-1] if len(self.data.volume) > 1 else 0

        if not self.position:
            # Buy Call: Break above ORB high with volume increase
            if (self.data.close[0] > self.orb_high) and (current_volume > prev_volume):
                self.order = self.buy()
                self.trade_state = 1
                # print(f"Buy CALL order opened on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

            # Buy Put: Break below ORB low with volume increase
            elif (self.data.close[0] < self.orb_low) and (current_volume > prev_volume):
                self.order = self.sell()  # Sell opens a short position for a put
                self.trade_state = -1
                # print(f"Buy PUT order opened on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

        elif self.trade_state == 1:  # In a long (call) position
            # Simple exit: close after one period (you can modify this)
            self.order = self.close()
            self.trade_state = 0
            # print(f"CALL position closed on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

        elif self.trade_state == -1:  # In a short (put) position
            self.order = self.close()
            self.trade_state = 0
            # print(f"PUT position closed on: {self.data.datetime.datetime(0)} at {self.data.close[0]}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                # print(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
                pass
            elif order.issell():
                # print(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
                pass
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # print(f"Order Canceled/Margin/Rejected: {order.status}")
            pass
        self.order = None