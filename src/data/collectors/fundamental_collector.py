#!/usr/bin/env python3
"""
NSE Fundamental Data Fetcher
Fetches fundamental data for all companies from the ticker CSV files
"""

import logging
import argparse
from pathlib import Path
import pandas as pd
from .fundamental_source import FundamentalSource


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fundamental_data_fetch.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main function to fetch fundamental data"""
    parser = argparse.ArgumentParser(description='Fetch NSE fundamental data for all companies')
    parser.add_argument('--batch-size', type=int, default=50, 
                       help='Number of requests before longer break (default: 50)')
    parser.add_argument('--delay', type=float, default=0.5, 
                       help='Delay between requests in seconds (default: 0.5)')
    parser.add_argument('--output', type=str, 
                       default=r"d:\QUANT\QT_python\quant-trading-platform\src\data\fundamental_data.csv",
                       help='Output CSV file path')
    parser.add_argument('--test', action='store_true', 
                       help='Test mode - fetch only first 10 companies')
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        # Initialize fundamental source
        logging.info("Initializing Fundamental Data Source...")
        fund_source = FundamentalSource()
        
        if args.test:
            logging.info("Running in TEST MODE - fetching first 10 companies only")
            # Load tickers and limit to first 10
            tickers = fund_source.load_tickers()[:10]
            
            # Manually fetch for test mode
            fundamental_data = []
            for i, ticker in enumerate(tickers):
                symbol = ticker['symbol']
                logging.info(f"Testing {i+1}/10: {symbol}")
                
                quote_data = fund_source.fetch_quote_data(symbol)
                if quote_data:
                    parsed_data = fund_source.parse_fundamental_data(symbol, quote_data)
                    parsed_data.update({
                        'market_type': ticker['market'],
                        'original_company_name': ticker['company_name']
                    })
                    fundamental_data.append(parsed_data)
                
                import time
                time.sleep(args.delay)
            
            df = pd.DataFrame(fundamental_data)
            test_output = args.output.replace('.csv', '_test.csv')
            fund_source.save_fundamentals(df, test_output)
            
            logging.info(f"Test completed. Data saved to {test_output}")
            print(f"\nTest Results:")
            print(f"Fetched data for {len(df)} companies")
            print(f"Sample data:")
            print(df[['symbol', 'company_name', 'last_price', 'market_cap', 'pe_ratio']].head())
            
        else:
            # Full fetch
            logging.info("Starting full fundamental data fetch...")
            output_path = fund_source.fetch_and_save_all(
                output_path=args.output,
                batch_size=args.batch_size,
                delay=args.delay
            )
            
            # Generate summary statistics
            df = pd.read_csv(output_path)
            stats = fund_source.get_summary_stats(df)
            
            logging.info("Fundamental data fetch completed!")
            print(f"\n=== SUMMARY STATISTICS ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
            
            print(f"\nData saved to: {output_path}")
            
            # Show sample of successful fetches
            successful_df = df[~df.get('error', pd.Series()).notna()]
            if len(successful_df) > 0:
                print(f"\nSample of successful fetches:")
                display_cols = ['symbol', 'company_name', 'last_price', 'market_cap', 'pe_ratio', 'sector']
                available_cols = [col for col in display_cols if col in successful_df.columns]
                print(successful_df[available_cols].head(10).to_string(index=False))
    
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        print("\nProcess interrupted. Partial data may be available.")
    
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
