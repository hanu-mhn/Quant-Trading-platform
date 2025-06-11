"""
Data Loader Module.

This module provides comprehensive data loading capabilities for various data sources
including CSV files, databases, and API endpoints.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import os
import logging
from pathlib import Path

from ...config.config import Config


class DataLoader:
    """Comprehensive data loader for market data from various sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config_manager = Config()
        self.config = config or self.config_manager.data
        
        # Set data paths
        self.raw_data_path = Path(self.config.raw_data_path)
        self.processed_data_path = Path(self.config.processed_data_path)
        self.external_data_path = Path(self.config.external_data_path)
        
        # Create directories if they don't exist
        for path in [self.raw_data_path, self.processed_data_path, self.external_data_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def load_csv_data(
        self,
        file_path: Union[str, Path],
        symbol_column: str = 'symbol',
        date_column: str = 'date',
        parse_dates: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file with automatic formatting.
        
        Args:
            file_path: Path to CSV file
            symbol_column: Name of symbol column
            date_column: Name of date column
            parse_dates: Whether to parse dates automatically
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Formatted DataFrame
        """
        try:
            # Default CSV reading parameters
            csv_params = {
                'index_col': False,
                'low_memory': False,
                **kwargs
            }
            
            if parse_dates and date_column:
                csv_params['parse_dates'] = [date_column]
            
            # Load data
            df = pd.read_csv(file_path, **csv_params)
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Ensure required columns exist
            if symbol_column.lower() not in df.columns:
                # Try to find symbol column with different names
                symbol_candidates = ['ticker', 'stock', 'instrument', 'name']
                for candidate in symbol_candidates:
                    if candidate in df.columns:
                        df = df.rename(columns={candidate: symbol_column.lower()})
                        break
                else:
                    self.logger.warning(f"Symbol column '{symbol_column}' not found in data")
            
            if date_column.lower() not in df.columns and parse_dates:
                # Try to find date column with different names
                date_candidates = ['timestamp', 'datetime', 'time', 'trading_date']
                for candidate in date_candidates:
                    if candidate in df.columns:
                        df = df.rename(columns={candidate: date_column.lower()})
                        if not pd.api.types.is_datetime64_any_dtype(df[date_column.lower()]):
                            df[date_column.lower()] = pd.to_datetime(df[date_column.lower()])
                        break
                else:
                    self.logger.warning(f"Date column '{date_column}' not found in data")
            
            # Sort by date if date column exists
            if date_column.lower() in df.columns:
                df = df.sort_values(date_column.lower())
                df = df.reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV data from {file_path}: {e}")
            raise
    
    def load_nse_equity_data(self, data_type: str = 'main') -> pd.DataFrame:
        """
        Load NSE equity data from standard files.
        
        Args:
            data_type: Type of data ('main' or 'sme')
            
        Returns:
            NSE equity DataFrame
        """
        file_mapping = {
            'main': 'main_equity.csv',
            'sme': 'sme_equity.csv'
        }
        
        if data_type not in file_mapping:
            raise ValueError(f"Invalid data_type. Must be one of: {list(file_mapping.keys())}")
        
        file_path = self.raw_data_path / file_mapping[data_type]
        
        if not file_path.exists():
            raise FileNotFoundError(f"NSE equity file not found: {file_path}")
        
        # Load with NSE-specific column handling
        df = self.load_csv_data(
            file_path,
            symbol_column='SYMBOL',
            date_column=None,
            parse_dates=False
        )
        
        # Standardize NSE columns
        column_mapping = {
            'symbol': 'symbol',
            'name_of_company': 'company_name',
            'series': 'series',
            'date_of_listing': 'listing_date',
            'paid_up_value': 'paid_up_value',
            'market_lot': 'market_lot',
            'isin_number': 'isin',
            'face_value': 'face_value'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Parse listing date if available
        if 'listing_date' in df.columns:
            df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
        
        return df
    
    def load_fundamental_data(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load fundamental data for symbols.
        
        Args:
            symbol: Specific symbol to load (optional)
            
        Returns:
            Fundamental data DataFrame
        """
        fundamental_files = list(self.raw_data_path.glob('*fundamental*.csv'))
        
        if not fundamental_files:
            raise FileNotFoundError("No fundamental data files found")
        
        # Load the most recent fundamental file
        latest_file = max(fundamental_files, key=os.path.getctime)
        
        df = self.load_csv_data(
            latest_file,
            symbol_column='symbol',
            date_column='date',
            parse_dates=True
        )
        
        # Filter by symbol if specified
        if symbol:
            df = df[df['symbol'].str.upper() == symbol.upper()]
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from NSE equity data."""
        try:
            nse_data = self.load_nse_equity_data('main')
            symbols = nse_data['symbol'].unique().tolist()
            return sorted(symbols)
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return metrics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'date_range': None,
            'unique_symbols': None
        }
        
        # Date range analysis
        if 'date' in df.columns:
            metrics['date_range'] = {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'days': (df['date'].max() - df['date'].min()).days
            }
        
        # Symbol analysis
        if 'symbol' in df.columns:
            metrics['unique_symbols'] = df['symbol'].nunique()
        
        return metrics