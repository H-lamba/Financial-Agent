import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# FIX: Argument order must be (tickers, start_date, end_date)
def download_stock_data(tickers, start_date, end_date):
    """
    Fetches daily stock data for given tickers.
    """
    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]

    print(f"📉 Downloading data for: {tickers}...")
    
    try:
        df = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True 
        )
        
        if df.empty:
            print(f"❌ No data found for {tickers}. (Check dates or ticker spelling)")
            return None

        return df

    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

def save_to_csv(df, file_name):
    if not os.path.exists("data"):
        os.makedirs("data")
    path = f"data/{file_name}"
    try:
        df.to_csv(path)
        print(f"💾 Data saved successfully to: {path}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    # Test
    data = download_stock_data("NVDA", "2024-01-01", "2024-01-05")
    if data is not None:
        print(data.head())