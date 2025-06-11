#!/usr/bin/env python3
"""
Fundamental Data Analyzer
Analyzes the collected fundamental data and provides insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging


class FundamentalAnalyzer:
    def __init__(self, data_path: str = None):
        """Initialize with path to fundamental data CSV"""
        self.data_path = data_path or r"d:\QUANT\QT_python\quant-trading-platform\src\data\fundamental_data.csv"
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load fundamental data from CSV"""
        try:
            self.df = pd.read_csv(self.data_path)
            # Convert numeric columns
            numeric_cols = ['last_price', 'market_cap', 'pe_ratio', 'pb_ratio', 'volume', 
                          'dividend_yield', 'book_value', 'face_value']
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            logging.info(f"Loaded {len(self.df)} companies from {self.data_path}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def basic_stats(self):
        """Generate basic statistics"""
        if self.df.empty:
            return "No data available"
        
        stats = {
            'Total Companies': len(self.df),
            'Main Market': len(self.df[self.df.get('market_type', '') == 'main']),
            'SME Market': len(self.df[self.df.get('market_type', '') == 'sme']),
            'Companies with Prices': len(self.df[self.df['last_price'].notna() & (self.df['last_price'] > 0)]),
            'Companies with Market Cap': len(self.df[self.df['market_cap'].notna() & (self.df['market_cap'] > 0)]),
            'Companies with PE Ratio': len(self.df[self.df['pe_ratio'].notna() & (self.df['pe_ratio'] > 0)]),
        }
        
        return stats
    
    def sector_analysis(self):
        """Analyze by sectors"""
        if 'sector' not in self.df.columns or self.df.empty:
            return "Sector data not available"
        
        sector_stats = self.df.groupby('sector').agg({
            'symbol': 'count',
            'last_price': ['mean', 'median'],
            'market_cap': ['mean', 'sum'],
            'pe_ratio': 'mean',
            'pb_ratio': 'mean'
        }).round(2)
        
        sector_stats.columns = ['Count', 'Avg_Price', 'Median_Price', 'Avg_MarketCap', 
                               'Total_MarketCap', 'Avg_PE', 'Avg_PB']
        
        return sector_stats.sort_values('Count', ascending=False)
    
    def top_companies(self, by='market_cap', top_n=20):
        """Get top companies by specified metric"""
        if by not in self.df.columns or self.df.empty:
            return f"Column {by} not available"
        
        # Filter out invalid values
        valid_df = self.df[self.df[by].notna() & (self.df[by] > 0)]
        
        if len(valid_df) == 0:
            return f"No valid data for {by}"
        
        top_companies = valid_df.nlargest(top_n, by)[
            ['symbol', 'company_name', by, 'sector', 'last_price', 'market_type']
        ]
        
        return top_companies
    
    def valuation_analysis(self):
        """Analyze valuation metrics"""
        if self.df.empty:
            return "No data available"
        
        # Filter companies with valid PE and PB ratios
        valid_df = self.df[
            (self.df['pe_ratio'].notna()) & 
            (self.df['pe_ratio'] > 0) & 
            (self.df['pe_ratio'] < 100) &  # Filter extreme outliers
            (self.df['pb_ratio'].notna()) & 
            (self.df['pb_ratio'] > 0) &
            (self.df['pb_ratio'] < 50)   # Filter extreme outliers
        ]
        
        if len(valid_df) == 0:
            return "No valid valuation data available"
        
        valuation_stats = {
            'PE Ratio': {
                'Mean': valid_df['pe_ratio'].mean(),
                'Median': valid_df['pe_ratio'].median(),
                'Std': valid_df['pe_ratio'].std(),
                'Min': valid_df['pe_ratio'].min(),
                'Max': valid_df['pe_ratio'].max()
            },
            'PB Ratio': {
                'Mean': valid_df['pb_ratio'].mean(),
                'Median': valid_df['pb_ratio'].median(),
                'Std': valid_df['pb_ratio'].std(),
                'Min': valid_df['pb_ratio'].min(),
                'Max': valid_df['pb_ratio'].max()
            }
        }
        
        return pd.DataFrame(valuation_stats).round(2)
    
    def market_cap_distribution(self):
        """Analyze market cap distribution"""
        if 'market_cap' not in self.df.columns or self.df.empty:
            return "Market cap data not available"
        
        valid_df = self.df[self.df['market_cap'].notna() & (self.df['market_cap'] > 0)]
        
        if len(valid_df) == 0:
            return "No valid market cap data"
        
        # Define market cap categories (in crores)
        def categorize_market_cap(market_cap):
            if market_cap >= 100000:
                return "Large Cap (>1L Cr)"
            elif market_cap >= 30000:
                return "Mid Cap (30K-1L Cr)"
            elif market_cap >= 5000:
                return "Small Cap (5K-30K Cr)"
            else:
                return "Micro Cap (<5K Cr)"
        
        valid_df['market_cap_category'] = valid_df['market_cap'].apply(categorize_market_cap)
        
        distribution = valid_df['market_cap_category'].value_counts()
        return distribution
    
    def generate_report(self, output_path: str = None):
        """Generate comprehensive analysis report"""
        if output_path is None:
            output_path = Path(self.data_path).parent / "fundamental_analysis_report.txt"
        
        with open(output_path, 'w') as f:
            f.write("NSE FUNDAMENTAL DATA ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic Statistics
            f.write("BASIC STATISTICS\n")
            f.write("-" * 20 + "\n")
            basic_stats = self.basic_stats()
            for key, value in basic_stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Top companies by market cap
            f.write("TOP 20 COMPANIES BY MARKET CAP\n")
            f.write("-" * 35 + "\n")
            top_by_mcap = self.top_companies('market_cap', 20)
            f.write(top_by_mcap.to_string(index=False))
            f.write("\n\n")
            
            # Sector Analysis
            f.write("SECTOR ANALYSIS\n")
            f.write("-" * 15 + "\n")
            sector_analysis = self.sector_analysis()
            if isinstance(sector_analysis, pd.DataFrame):
                f.write(sector_analysis.to_string())
            else:
                f.write(str(sector_analysis))
            f.write("\n\n")
            
            # Valuation Analysis
            f.write("VALUATION ANALYSIS\n")
            f.write("-" * 18 + "\n")
            valuation_analysis = self.valuation_analysis()
            if isinstance(valuation_analysis, pd.DataFrame):
                f.write(valuation_analysis.to_string())
            else:
                f.write(str(valuation_analysis))
            f.write("\n\n")
            
            # Market Cap Distribution
            f.write("MARKET CAP DISTRIBUTION\n")
            f.write("-" * 23 + "\n")
            mcap_dist = self.market_cap_distribution()
            if isinstance(mcap_dist, pd.Series):
                f.write(mcap_dist.to_string())
            else:
                f.write(str(mcap_dist))
            f.write("\n\n")
        
        print(f"Analysis report saved to: {output_path}")
        return output_path


def main():
    """Main function for analysis"""
    logging.basicConfig(level=logging.INFO)
    
    analyzer = FundamentalAnalyzer()
    
    if analyzer.df.empty:
        print("No data to analyze. Please run fetch_fundamentals.py first.")
        return
    
    print("NSE FUNDAMENTAL DATA ANALYSIS")
    print("=" * 40)
    
    # Basic Statistics
    print("\nBASIC STATISTICS:")
    stats = analyzer.basic_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Top companies by market cap
    print(f"\nTOP 10 COMPANIES BY MARKET CAP:")
    top_companies = analyzer.top_companies('market_cap', 10)
    if isinstance(top_companies, pd.DataFrame):
        print(top_companies[['symbol', 'company_name', 'market_cap', 'sector']].to_string(index=False))
    
    # Market cap distribution
    print(f"\nMARKET CAP DISTRIBUTION:")
    mcap_dist = analyzer.market_cap_distribution()
    if isinstance(mcap_dist, pd.Series):
        for category, count in mcap_dist.items():
            print(f"  {category}: {count} companies")
    
    # Generate full report
    report_path = analyzer.generate_report()
    print(f"\nFull analysis report saved to: {report_path}")


if __name__ == "__main__":
    main()
