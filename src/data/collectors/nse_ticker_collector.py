import pandas as pd
import requests
from pathlib import Path
import logging
from io import StringIO
import os
import json
import time
from urllib.parse import urlencode

def get_nse_session() -> requests.Session:
    """Create a session with required NSE headers and cookies"""
    session = requests.Session()
    
    # Initial headers for the homepage request
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
        
        return session
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to initialize NSE session: {str(e)}")
        raise

def _fetch_data_from_url(url: str, session: requests.Session) -> pd.DataFrame:
    """Helper function to fetch and parse data from a URL"""
    logging.info(f"Fetching data from {url}")
    try:
        # For NSE archives, try with a fresh session and specific headers
        if 'nsearchives.nseindia.com' in url:
            # Create a new session specifically for archives
            archive_session = requests.Session()
            archive_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebP/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            archive_session.headers.update(archive_headers)
            response = archive_session.get(url, timeout=15, allow_redirects=True)
        else:
            response = session.get(url, timeout=10)
        
        response.raise_for_status()
        
        # Check if this is a CSV file
        if url.endswith('.csv'):
            # Handle CSV response
            df = pd.read_csv(StringIO(response.text))
            logging.info(f"Available columns: {df.columns.tolist()}")
            return df
        else:
            # Handle JSON response
            try:
                data = response.json()
            except Exception as e:
                logging.error(f"Response not JSON. Raw response: {response.text[:500]}")
                raise e
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame(data)
            logging.info(f"Available columns: {df.columns.tolist()}")
            return df
    except Exception as e:
        logging.error(f"Error fetching data from {url}: {e}")
        return pd.DataFrame(columns=["Symbol", "Company Name"])

def fetch_nse_equity_tickers(
    main_output: str = r"d:\QUANT\QT_python\quant-trading-platform\src\data\main_equity.csv",
    sme_output: str = r"d:\QUANT\QT_python\quant-trading-platform\src\data\sme_equity.csv",
    equity_url: str = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    sme_url: str = "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv",
    return_df: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fetch NSE equity tickers using authenticated session"""
    session = get_nse_session()
    df_main = _fetch_data_from_url(equity_url, session)
    time.sleep(2)
    df_sme = _fetch_data_from_url(sme_url, session)    # Process main equity data
    if not df_main.empty:
        # For CSV files, handle the actual column names from NSE archives
        if 'SYMBOL' in df_main.columns and 'NAME OF COMPANY' in df_main.columns:
            df_main = df_main[['SYMBOL', 'NAME OF COMPANY']].rename(columns={'SYMBOL': 'Symbol', 'NAME OF COMPANY': 'Company Name'})
        elif 'symbol' in df_main.columns and 'companyName' in df_main.columns:
            df_main = df_main[['symbol', 'companyName']].rename(columns={'symbol': 'Symbol', 'companyName': 'Company Name'})
        elif 'data' in df_main:
            symbols = []
            company_names = []
            for item in df_main['data']:
                symbols.append(item.get('symbol', ''))
                company_names.append(item.get('companyName', ''))
            df_main = pd.DataFrame({'Symbol': symbols, 'Company Name': company_names})
        
        df_main = df_main.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"])
        main_path = Path(main_output)
        main_path.parent.mkdir(parents=True, exist_ok=True)
        df_main.to_csv(main_path, index=False)
        logging.info(f"Saved {len(df_main)} main equity tickers to {main_path}")
      # Process SME data
    if not df_sme.empty:
        # For CSV files, handle the actual column names from NSE archives
        if 'SYMBOL' in df_sme.columns and 'NAME OF COMPANY' in df_sme.columns:
            df_sme = df_sme[['SYMBOL', 'NAME OF COMPANY']].rename(columns={'SYMBOL': 'Symbol', 'NAME OF COMPANY': 'Company Name'})
        elif 'SYMBOL' in df_sme.columns and 'NAME_OF_COMPANY' in df_sme.columns:
            df_sme = df_sme[['SYMBOL', 'NAME_OF_COMPANY']].rename(columns={'SYMBOL': 'Symbol', 'NAME_OF_COMPANY': 'Company Name'})
        elif 'symbol' in df_sme.columns and 'companyName' in df_sme.columns:
            df_sme = df_sme[['symbol', 'companyName']].rename(columns={'symbol': 'Symbol', 'companyName': 'Company Name'})
        elif 'data' in df_sme:
            symbols = []
            company_names = []
            for item in df_sme['data']:
                symbols.append(item.get('symbol', ''))
                company_names.append(item.get('companyName', ''))
            df_sme = pd.DataFrame({'Symbol': symbols, 'Company Name': company_names})
        
        df_sme = df_sme.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"])
        sme_path = Path(sme_output)
        sme_path.parent.mkdir(parents=True, exist_ok=True)
        df_sme.to_csv(sme_path, index=False)
        logging.info(f"Saved {len(df_sme)} SME tickers to {sme_path}")
    if return_df:
        return df_main, df_sme

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_nse_equity_tickers(return_df=False)
    logging.info("NSE equity and SME tickers fetched and saved separately.")
