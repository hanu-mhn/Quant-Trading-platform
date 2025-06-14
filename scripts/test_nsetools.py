#!/usr/bin/env python3
"""
Test nsetools functionality
"""

try:
    from nsetools import Nse
    import time
    
    print("Testing nsetools...")
    nse = Nse()
    
    # Test basic functionality
    print("NSE object created successfully")
    
    # Test with a popular stock
    test_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    
    for symbol in test_symbols:
        try:
            print(f"Testing {symbol}...")
            quote = nse.get_quote(symbol)
            if quote:
                print(f"  Success: {quote.get('companyName', 'N/A')} - Price: {quote.get('lastPrice', 'N/A')}")
            else:
                print(f"  No data returned for {symbol}")
            time.sleep(1)
        except Exception as e:
            print(f"  Error with {symbol}: {e}")
    
    print("nsetools test completed")

except ImportError as e:
    print(f"nsetools not available: {e}")
except Exception as e:
    print(f"Error testing nsetools: {e}")
