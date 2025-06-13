import unittest
from unittest.mock import Mock, patch
from src.live_trading.live_trader import LiveTrader
from src.live_trading.brokers.example_broker import ExampleBroker

class TestLiveTrader(unittest.TestCase):

    def setUp(self):
        self.broker = ExampleBroker()
        self.trader = LiveTrader(self.broker)

    def test_start_trading(self):
        self.broker.connect = Mock(return_value=True)
        result = self.trader.start_trading()
        self.assertTrue(result)
        self.broker.connect.assert_called_once()

    def test_stop_trading(self):
        # Set up the trader as if it were running
        self.trader.is_running = True
        self.broker.disconnect = Mock(return_value=True)
        
        # Call the method under test
        self.trader.stop_trading()
        
        # Verify disconnect was called and state was updated
        self.broker.disconnect.assert_called_once()
        self.assertFalse(self.trader.is_running)

if __name__ == '__main__':
    unittest.main()