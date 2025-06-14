import requests
import pandas as pd
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
from io import StringIO


class FundamentalSource:
    def __init__(self, main_equity_path: str = None, sme_equity_path: str = None):
        """Initialize the fundamental data source with ticker CSV paths"""
        self.main_equity_path = main_equity_path or r"d:\QUANT\QT_python\quant-trading-platform\src\data\main_equity.csv"
        self.sme_equity_path = sme_equity_path or r"d:\QUANT\QT_python\quant-trading-platform\src\data\sme_equity.csv"
        
        # NSE session setup
        self.session = self._create_session()
          # Output path for fundamental data
        self.output_path = Path(r"d:\QUANT\QT_python\quant-trading-platform\src\data\fundamental_data.csv")
        
    def _create_session(self):
        """Create an authenticated NSE session"""
        session = requests.Session()
        base_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"'
        }
        session.headers.update(base_headers)
        
        try:
            # Initial visit to homepage to get cookies
            home_url = 'https://www.nseindia.com'
            home = session.get(home_url, timeout=10, allow_redirects=True)
            home.raise_for_status()
            time.sleep(1)
            
            # Visit the market data page to set additional cookies
            market_url = f'{home_url}/market-data/live-equity-market'
            market = session.get(market_url, timeout=10)
            market.raise_for_status()
            time.sleep(2)
            
            # Update headers for API requests
            api_headers = {
                'Accept': 'application/json, text/plain, */*',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': market_url,
                'Host': 'www.nseindia.com',
                'Origin': 'https://www.nseindia.com'
            }
            session.headers.update(api_headers)
            
            logging.info("NSE session initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize NSE session: {e}")
            
        return session

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
            """Fetch quote and fundamental data for a single symbol"""
            try:
                # Try multiple API endpoints
                urls = [
                    f"https://www.nseindia.com/api/quote-equity?symbol={symbol}",
                    f"https://www.nseindia.com/api/quote-equity?symbol={symbol}&section=trade_info",
                    f"https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500&symbol={symbol}"
                ]
                
                for url in urls:
                    try:
                        response = self.session.get(url, timeout=10)
                        response.raise_for_status()
                        
                        # Debug: log response content
                        if len(response.text.strip()) == 0:
                            logging.debug(f"Empty response for {symbol} from {url}")
                            continue
                        
                        # Try to parse JSON
                        data = response.json()
                        if data and isinstance(data, dict):
                            logging.debug(f"Successfully fetched data for {symbol}")
                            return data
                        else:
                            logging.debug(f"Invalid data structure for {symbol} from {url}")
                            continue
                            
                    except requests.exceptions.HTTPError as e:
                        logging.debug(f"HTTP error for {symbol} from {url}: {e}")
                        continue
                    except json.JSONDecodeError as e:
                        logging.debug(f"JSON decode error for {symbol} from {url}: Response: {response.text[:200]}")
                        continue
                
                # If all URLs fail, try using a simpler approach with market data
                return self._fetch_basic_market_data(symbol)
                
            except Exception as e:
                logging.warning(f"Error fetching data for {symbol}: {e}")
                return None
    
    def _fetch_basic_market_data(self, symbol: str) -> Optional[Dict]:
        """Fallback method to fetch basic market data"""
        try:
            # Try the market data API
            url = f"https://www.nseindia.com/api/marketStatus"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # This is a basic fallback - we'll return minimal data
            return {
                'info': {'companyName': f'{symbol} Limited'},
                'priceInfo': {'lastPrice': 0, 'change': 0, 'pChange': 0},
                'securityInfo': {'isin': '', 'series': 'EQ'}
            }
            
        except Exception as e:
            logging.debug(f"Fallback method also failed for {symbol}: {e}")
            return None

    def parse_fundamental_data(self, symbol: str, raw_data: Dict) -> Dict:
        """Parse fundamental data from NSE quote response"""
        try:
            info = raw_data.get('info', {})
            price_info = raw_data.get('priceInfo', {})
            security_info = raw_data.get('securityInfo', {})
            
            # Extract fundamental metrics
            fundamental_data = {
                'symbol': symbol,
                'company_name': info.get('companyName', ''),
                'industry': info.get('industry', ''),
                'sector': info.get('sector', ''),
                'isin': security_info.get('isin', ''),
                'series': security_info.get('series', ''),
                
                # Price data
                'last_price': price_info.get('lastPrice', 0),
                'change': price_info.get('change', 0),
                'change_percent': price_info.get('pChange', 0),
                'open': price_info.get('open', 0),
                'high': price_info.get('intraDayHighLow', {}).get('max', 0),
                'low': price_info.get('intraDayHighLow', {}).get('min', 0),
                'close': price_info.get('close', 0),
                'volume': price_info.get('totalTradedVolume', 0),
                'value': price_info.get('totalTradedValue', 0),
                
                # 52-week data
                'week_52_high': price_info.get('weekHighLow', {}).get('max', 0),
                'week_52_low': price_info.get('weekHighLow', {}).get('min', 0),
                
                # Fundamental ratios (if available)
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('pdSectorPe', 0),
                'pb_ratio': info.get('pdSectorPb', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'book_value': info.get('bookValue', 0),
                'face_value': info.get('faceValue', 0),
                
                # Additional metrics
                'listing_date': info.get('listingDate', ''),
                'last_update_time': price_info.get('lastUpdateTime', ''),
            }
            
            return fundamental_data
            
        except Exception as e:
            logging.error(f"Error parsing fundamental data for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def fetch_all_fundamentals(self, batch_size: int = 50, delay: float = 0.5) -> pd.DataFrame:
        """Fetch fundamental data for all companies"""
        tickers = self.load_tickers()
        fundamental_data = []
        
        logging.info(f"Starting to fetch fundamental data for {len(tickers)} companies")
        
        for i, ticker in enumerate(tickers):
            symbol = ticker['symbol']
            
            try:
                # Fetch quote data
                quote_data = self.fetch_quote_data(symbol)
                
                if quote_data:
                    # Parse fundamental data
                    parsed_data = self.parse_fundamental_data(symbol, quote_data)
                    parsed_data.update({
                        'market_type': ticker['market'],
                        'original_company_name': ticker['company_name']
                    })
                    fundamental_data.append(parsed_data)
                    
                    logging.info(f"Processed {i+1}/{len(tickers)}: {symbol}")
                else:
                    # Add empty record for failed fetches
                    fundamental_data.append({
                        'symbol': symbol,
                        'market_type': ticker['market'],
                        'original_company_name': ticker['company_name'],
                        'error': 'Failed to fetch data'
                    })
                
                # Rate limiting
                if (i + 1) % batch_size == 0:
                    logging.info(f"Processed {i+1} companies, taking a longer break...")
                    time.sleep(delay * 5)  # Longer break after batch
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

    def fetch_and_save_all(self, output_path: str = None, batch_size: int = 50, delay: float = 0.5) -> str:
        """Fetch all fundamental data and save to CSV"""
        df = self.fetch_all_fundamentals(batch_size=batch_size, delay=delay)
        return self.save_fundamentals(df, output_path)

    def get_summary_stats(self, df: pd.DataFrame = None) -> Dict:
        """Get summary statistics of the fundamental data"""
        if df is None:
            try:
                df = pd.read_csv(self.output_path)
            except:
                return {"error": "No data available"}
        
        stats = {
            'total_companies': len(df),
            'main_market': len(df[df['market_type'] == 'main']) if 'market_type' in df.columns else 0,
            'sme_market': len(df[df['market_type'] == 'sme']) if 'market_type' in df.columns else 0,
            'successful_fetches': len(df[~df.get('error', pd.Series()).notna()]),
            'failed_fetches': len(df[df.get('error', pd.Series()).notna()]),
        }
        
        # Add numerical statistics if available
        numeric_columns = ['last_price', 'market_cap', 'pe_ratio', 'pb_ratio', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                valid_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(valid_data) > 0:
                    stats[f'{col}_mean'] = valid_data.mean()
                    stats[f'{col}_median'] = valid_data.median()
                    stats[f'{col}_std'] = valid_data.std()
        
        return stats