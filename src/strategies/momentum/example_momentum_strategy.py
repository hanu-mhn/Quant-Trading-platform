class ExampleStrategy:
    def __init__(self):
        self.initialized = False

    def initialize(self):
        # Setup parameters and indicators for the strategy
        self.initialized = True

    def execute(self, market_data):
        if not self.initialized:
            raise Exception("Strategy not initialized. Call initialize() first.")
        
        # Implement the trading logic here
        # Example: Buy if the price is below a certain threshold
        if market_data['price'] < 100:
            return "buy"
        elif market_data['price'] > 150:
            return "sell"
        else:
            return "hold"