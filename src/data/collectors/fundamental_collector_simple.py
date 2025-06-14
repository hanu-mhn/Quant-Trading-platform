#!/usr/bin/env python3
"""
Simple NSE Fundamental Data Fetcher using nsetools
Fetches basic fundamental data for all companies from the ticker CSV files
"""

import logging
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Optional
from nsetools import Nse


class SimpleFundamentalSource:
    def __init__(self, main_equity_path: str = None, sme_equity_path: str = None):
        """Initialize the fundamental data source with ticker CSV paths"""
        self.main_equity_path = main_equity_path or r"d:\QUANT\QT_python\quant-trading-platform\src\data\main_equity.csv"
        self.sme_equity_path = sme_equity_path or r"d:\QUANT\QT_python\quant-trading-platform\src\data\sme_equity.csv"
        
        # Initialize NSE object
        self.nse = Nse()
        
        # Output path for fundamental data
        self.output_path = Path(r"d:\QUANT\QT_python\quant-trading-platform\src\data\fundamental_data_simple.csv")
        
    def load_tickers(self) -> List[Dict]:
        """Load all tickers from CSV files"""
        tickers = []
        
        # Load main equity tickers
        try:
            main_df = pd.read_csv(self.main_equity_path)
            for _, row in main_df.iterrows():
                tickers.append({
                    'symbol': row['Symbol'],
                    'company_name': row['Company Name'],
                    'market': 'main'
                })
            logging.info(f"Loaded {len(main_df)} main equity tickers")
        except Exception as e:
            logging.error(f"Error loading main equity tickers: {e}")
        
        # Load SME tickers
        try:
            sme_df = pd.read_csv(self.sme_equity_path)
            for _, row in sme_df.iterrows():
                tickers.append({
                    'symbol': row['Symbol'],
                    'company_name': row['Company Name'],
                    'market': 'sme'
                })
            logging.info(f"Loaded {len(sme_df)} SME tickers")
        except Exception as e:
            logging.error(f"Error loading SME tickers: {e}")
            
        return tickers

    def fetch_quote_data(self, symbol: str) -> Optional[Dict]:
        """Fetch quote data for a single symbol using nsetools"""
        try:
            # Get quote data
            quote = self.nse.get_quote(symbol)
            return quote
            
        except Exception as e:
            logging.debug(f"Error fetching data for {symbol}: {e}")
            return None

    def parse_fundamental_data(self, symbol: str, raw_data: Dict) -> Dict:
        """Parse fundamental data from nsetools response"""
        try:
            if not raw_data:
                return {'symbol': symbol, 'error': 'No data available'}
            
            # Extract fundamental metrics
            fundamental_data = {
                'symbol': symbol,
                'company_name': raw_data.get('companyName', ''),
                'industry': raw_data.get('industry', ''),
                'series': raw_data.get('series', ''),
                'isin': raw_data.get('isinCode', ''),
                
                # Price data
                'last_price': self._safe_float(raw_data.get('lastPrice')),
                'change': self._safe_float(raw_data.get('change')),
                'change_percent': self._safe_float(raw_data.get('pChange')),
                'open': self._safe_float(raw_data.get('open')),
                'high': self._safe_float(raw_data.get('dayHigh')),
                'low': self._safe_float(raw_data.get('dayLow')),
                'close': self._safe_float(raw_data.get('previousClose')),
                'volume': self._safe_float(raw_data.get('totalTradedVolume')),
                'value': self._safe_float(raw_data.get('totalTradedValue')),
                
                # 52-week data
                'week_52_high': self._safe_float(raw_data.get('high52')),
                'week_52_low': self._safe_float(raw_data.get('low52')),
                
                # Fundamental ratios
                'market_cap': self._safe_float(raw_data.get('marketCap')),
                'pe_ratio': self._safe_float(raw_data.get('pe')),
                'eps': self._safe_float(raw_data.get('eps')),
                'face_value': self._safe_float(raw_data.get('faceValue')),
                'book_value': self._safe_float(raw_data.get('bookValue')),
                
                # Additional info
                'sector': raw_data.get('sector', ''),
                'listing_date': raw_data.get('listingDate', ''),
            }
            
            return fundamental_data
            
        except Exception as e:
            logging.error(f"Error parsing fundamental data for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def _safe_float(self, value):
        """Safely convert value to float"""
        try:
            if value is None or value == '' or value == '-':
                return 0.0
            return float(str(value).replace(',', ''))
        except:
            return 0.0

    def fetch_all_fundamentals(self, batch_size: int = 50, delay: float = 1.0) -> pd.DataFrame:
        """Fetch fundamental data for all companies"""
        tickers = self.load_tickers()
        fundamental_data = []
        
        logging.info(f"Starting to fetch fundamental data for {len(tickers)} companies using nsetools")
        
        for i, ticker in enumerate(tickers):
            symbol = ticker['symbol']
            
            try:
                # Fetch quote data
                quote_data = self.fetch_quote_data(symbol)
                
                # Parse fundamental data
                parsed_data = self.parse_fundamental_data(symbol, quote_data)
                parsed_data.update({
                    'market_type': ticker['market'],
                    'original_company_name': ticker['company_name']
                })
                fundamental_data.append(parsed_data)
                
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i+1}/{len(tickers)}: {symbol}")
                
                # Rate limiting
                if (i + 1) % batch_size == 0:
                    logging.info(f"Processed {i+1} companies, taking a longer break...")
                    time.sleep(delay * 3)
                else:
                    time.sleep(delay)
                    
            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")
                fundamental_data.append({
                    'symbol': symbol,
                    'market_type': ticker['market'],
                    'original_company_name': ticker['company_name'],
                    'error': str(e)
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(fundamental_data)
        return df

    def save_fundamentals(self, df: pd.DataFrame, output_path: str = None) -> str:
        """Save fundamental data to CSV"""
        if output_path is None:
            output_path = self.output_path
        else:
            output_path = Path(output_path)
            
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logging.info(f"Saved fundamental data for {len(df)} companies to {output_path}")
        
        return str(output_path)

    def fetch_and_save_all(self, output_path: str = None, batch_size: int = 50, delay: float = 1.0) -> str:
        """Fetch all fundamental data and save to CSV"""
        df = self.fetch_all_fundamentals(batch_size=batch_size, delay=delay)
        return self.save_fundamentals(df, output_path)


def main():
    """Main function to fetch fundamental data using nsetools"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch NSE fundamental data using nsetools')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--delay', type=float, default=1.0)
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        fund_source = SimpleFundamentalSource()
        
        if args.test:
            logging.info("Running in TEST MODE - fetching first 5 companies only")
            tickers = fund_source.load_tickers()[:5]
            
            fundamental_data = []
            for i, ticker in enumerate(tickers):
                symbol = ticker['symbol']
                logging.info(f"Testing {i+1}/5: {symbol}")
                
                quote_data = fund_source.fetch_quote_data(symbol)
                parsed_data = fund_source.parse_fundamental_data(symbol, quote_data)
                parsed_data.update({
                    'market_type': ticker['market'],
                    'original_company_name': ticker['company_name']
                })
                fundamental_data.append(parsed_data)
                time.sleep(args.delay)
            
            df = pd.DataFrame(fundamental_data)
            test_output = str(fund_source.output_path).replace('.csv', '_test.csv')
            fund_source.save_fundamentals(df, test_output)
            
            print(f"\nTest Results:")
            print(f"Fetched data for {len(df)} companies")
            successful_df = df[~df.get('error', pd.Series()).notna()]
            print(f"Successful fetches: {len(successful_df)}")
            
            if len(successful_df) > 0:
                print(f"\nSample successful data:")
                display_cols = ['symbol', 'company_name', 'last_price', 'market_cap', 'pe_ratio']
                available_cols = [col for col in display_cols if col in successful_df.columns]
                print(successful_df[available_cols].head().to_string(index=False))
        else:
            # Full fetch
            output_path = fund_source.fetch_and_save_all(
                batch_size=args.batch_size,
                delay=args.delay
            )
            print(f"Data saved to: {output_path}")
            
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
