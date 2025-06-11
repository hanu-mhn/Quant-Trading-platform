import unittest
from src.backtesting.backtest import Backtester
from src.backtesting.strategies.example_strategy import ExampleStrategy

class TestBacktester(unittest.TestCase):

    def setUp(self):
        self.strategy = ExampleStrategy()
        self.backtester = Backtester(self.strategy)

    def test_run_backtest(self):
        results = self.backtester.run_backtest()
        self.assertIsNotNone(results)
        self.assertIn('performance', results)

    def test_analyze_results(self):
        results = self.backtester.run_backtest()
        analysis = self.backtester.analyze_results(results)
        self.assertIsNotNone(analysis)
        self.assertIn('summary', analysis)

if __name__ == '__main__':
    unittest.main()