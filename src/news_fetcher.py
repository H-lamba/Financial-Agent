import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<.*?>', '', text)
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def fetch_financial_news(ticker, days=30):
    if not NEWS_API_KEY:
        print("Error with API")
        return []
    print(f"fetching news regarding {ticker}....")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} stock",
        "language":"en",
        "sortBy" : "relevancy",
        "apiKey": NEWS_API_KEY,
        "pageSize": 10
    }
    
    try:
        response = requests.get(url, params= params)
        data = response.json()
        
        if(data.get("status")!= "ok"):
            print("Error occured cannot fetch the data")
            return []
        articles = data.get("articles",[])
        clean_news = []
        for art in articles:
            clean_news.append({
                "date" : art.get("publishedAt")[:10],
                "title": clean_text(art.get("title")),
                "description": clean_text(art.get("description")),
                "url": art.get("url")
            })
        print("Found..... articles")
        return clean_news

    except Exception as e:
        print(f"AN unknown error occured {e}")
        return []
    
if __name__ == "__main__":
    # Test with NVDA
    d = int(input("Enter the number of days: "))
    news = fetch_financial_news("NVDA",d)
    
    # Print the first headline found
    if news:
        print("\n--- Top Headline ---")
        print(f"Title: {news[0]['title']}")
        print(f"Summary: {news[0]['description']}")
    