import pandas as pd
import os

def load_tickers(file_path="d:/QUANT/QT_python/quant-trading-platform/src/data/tickers.csv"):
    """
    Load tickers from the provided CSV file.
    """
    try:
        # Print the current working directory for debugging
        print("Current Working Directory:", os.getcwd())

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Ensure the file has the expected columns
        if "Symbol" not in df.columns or "Company Name" not in df.columns:
            raise ValueError("The file must contain 'Symbol' and 'Company Name' columns.")

        # Extract the tickers (Symbol column)
        tickers = df["Symbol"].tolist()

        print(f"Successfully loaded {len(tickers)} tickers.")
        return tickers
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except ValueError as e:
        print(f"Error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

if __name__ == "__main__":
    tickers = load_tickers()
    print("Tickers:", tickers)