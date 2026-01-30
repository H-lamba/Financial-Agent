import pandas as pd
from data_loader import download_stock_data, save_to_csv
from news_fetcher import fetch_financial_news
from sentiment import get_sentiment  # ← NEW: Import sentiment function

def get_merged_data(ticker, startdate, enddate):
    """
    Fetch stock data and news, then merge with sentiment analysis.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'NVDA')
        startdate (str): Start date 'YYYY-MM-DD'
        enddate (str): End date 'YYYY-MM-DD'
        
    Returns:
        pd.DataFrame: Merged data with columns:
            - date, open, high, low, close, volume
            - title, description (lists of news for each day)
            - sentiment_score (NEW: -1 to +1)
    """
    # Get stock data
    stock_df = download_stock_data(ticker,startdate, enddate)
    
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(-1)
    
    stock_df = stock_df.reset_index()
    stock_df.columns = [c.lower() for c in stock_df.columns]
    stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date
    
    # Get news data
    news_list = fetch_financial_news(ticker, 30)  # Increased from 10 to 30 days
    news_df = pd.DataFrame(news_list)
    
    if news_df.empty:
        print("⚠️  No news found")
        stock_df["title"] = None
        stock_df["description"] = None
        stock_df["sentiment_score"] = 0.0  # ← NEW: Add sentiment column
        return stock_df
    
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
    
    # Group news by date
    daily_news = news_df.groupby("date").agg({
        "title": lambda x: list(x),
        "description": lambda x: list(x)
    }).reset_index()
    
    # ========================================================================
    # NEW: Calculate sentiment for each day's news
    # ========================================================================
    
    print(f"📊 Analyzing sentiment for {len(daily_news)} days...")
    
    def calculate_sentiment(row):
        """Calculate sentiment score for a day's news."""
        titles = row['title'] if isinstance(row['title'], list) else []
        descriptions = row['description'] if isinstance(row['description'], list) else []
        
        # Combine title + description for each article
        texts = []
        for i in range(max(len(titles), len(descriptions))):
            title = titles[i] if i < len(titles) else ''
            desc = descriptions[i] if i < len(descriptions) else ''
            
            # Combine (title is weighted more by being first)
            if title and desc:
                combined = f"{title}. {desc}"
            else:
                combined = title or desc
            
            if combined.strip():
                texts.append(combined)
        
        # Calculate sentiment with confidence weighting
        if texts:
            return get_sentiment(
                texts,
                confidence_threshold=0.0,  # Use all predictions
                use_weighting=True,        # Weight by confidence
                verbose=False
            )
        else:
            return 0.0
    
    # Apply sentiment calculation to each day
    daily_news['sentiment_score'] = daily_news.apply(calculate_sentiment, axis=1)
    
    # ========================================================================
    
    # Merge stock data with news and sentiment
    merged = pd.merge(stock_df, daily_news, on="date", how="left")
    
    # Fill missing values (days with no news)
    merged['sentiment_score'] = merged['sentiment_score'].fillna(0.0)
    
    print(f"✅ Sentiment analysis complete!")
    print(f"   Average sentiment: {merged['sentiment_score'].mean():.3f}")
    
    return merged


if __name__ == "__main__":
    # Test with a short range
    print("=" * 70)
    print("Testing process_data.py with Sentiment Analysis")
    print("=" * 70)
    
    ticker = "ITC.NS" 
    
    print(f"🚀 Running Analysis for {ticker}...")
    df = get_merged_data(ticker, "2025-12-01", "2026-02-01")
    # Print columns to verify
    print("\n📋 Columns:", list(df.columns))
    
    # Print first few rows
    print("\n📊 First 5 rows:")
    print(df[['date', 'close', 'sentiment_score']].head())
    
    # Print summary statistics
    print("\n📈 Sentiment Statistics:")
    print(f"   Mean:    {df['sentiment_score'].mean():+.3f}")
    print(f"   Std Dev: {df['sentiment_score'].std():.3f}")
    print(f"   Min:     {df['sentiment_score'].min():+.3f}")
    print(f"   Max:     {df['sentiment_score'].max():+.3f}")
    
    # Save to CSV
    output_filename = f"{ticker}_with_sentiment.csv"
    save_to_csv(df, output_filename)
    
    print("\n" + "=" * 70)
    print(f"✅ Test complete! Saved to data/{output_filename}")
    print("=" * 70)
