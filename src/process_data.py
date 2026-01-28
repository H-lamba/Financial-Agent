import pandas as pd
from data_loader import download_stock_data, save_to_csv
from news_fetcher import fetch_financial_news

def get_merged_data(ticker, startdate, endate):
    stock_df = download_stock_data(startdate,endate, ticker)
    if isinstance(stock_df.columns, pd.MultiIndex):
        # We only want the last level ('Open', 'Close', etc.)
        stock_df.columns = stock_df.columns.get_level_values(-1)
    stock_df = stock_df.reset_index()
    stock_df.columns = [c.lower() for c in stock_df.columns]
    
    stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date
    news_list = fetch_financial_news(ticker, 10)
    news_df = pd.DataFrame(news_list)
    
    if news_df.empty:
        print("No new found")
        stock_df["title"] = None
        stock_df["description"] = None
        return stock_df
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
    
    
    daily_news = news_df.groupby("date").agg({
        "title": lambda x:list(x),
        "description": lambda x: list(x)
    }).reset_index()
    
    merged = pd.merge(stock_df, daily_news, on = "date", how = "left")

    return merged

if __name__ == "__main__":
    # Test with a short range
    df = get_merged_data("NVDA", "2026-01-01", "2026-01-15")
    df = get_merged_data("NVDA", "2026-01-01", "2026-01-15")
    # Print columns to verify they are clean strings now
    print("Columns:", df.columns)
    print(df.head())