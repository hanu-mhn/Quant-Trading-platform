import unittest
from src.live_trading.live_trader import LiveTrader
from src.live_trading.brokers.example_broker import ExampleBroker

class TestLiveTrader(unittest.TestCase):

    def setUp(self):
        self.broker = ExampleBroker()
        self.trader = LiveTrader(self.broker)

    def test_start_trading(self):
        self.broker.connect = unittest.mock.Mock(return_value=True)
        result = self.trader.start_trading()
        self.assertTrue(result)
        self.broker.connect.assert_called_once()

    def test_stop_trading(self):
        self.trader.stop_trading()
        # Assuming stop_trading does not return anything, we can check if it completes without error
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()