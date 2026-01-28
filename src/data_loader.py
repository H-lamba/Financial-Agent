from datetime import datetime
import matplotlib.pyplot as plt
##import seaborn as sns
import yfinance as yf
import os

def download_stock_data(startdate, enddate, tickers):
    df = yf.download(
        tickers = tickers,
        start= startdate,
        end = enddate,
        interval = "1d",
        group_by= "ticker",
        auto_adjust= True,
        progress= False,
        threads= False 
    )
    if df.empty:
        print("❌ No data found. Check your dates or tickers.")
        return None
    ##df.to_csv(file_path, mode='a', header=False, index=False)

    return df
def save_to_csv(df, file_name):
    """
    Saves the DataFrame to a CSV file in the 'data' folder.
    """
    # 1. Create the 'data' folder if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # 2. Define the full path
    path = f"data/{file_name}"
    
    # 3. Save
    try:
        df.to_csv(path)
        print(f"💾 Data saved successfully to: {path}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        
if __name__ == "__main__":
    end_date = datetime.now()
    date_str = input("Enter the date (YYYY-MM-DD): ")

    start_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    stocks = "NVDA"
    df = download_stock_data(start_date,end_date, stocks)
    if df is not None and not df.empty:
        # We create a filename like "NVDA_stock_data.csv"
        filename = f"{stocks}_stock_data.csv"
        save_to_csv(df, filename)