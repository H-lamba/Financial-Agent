import os 
from dotenv import load_dotenv
import yfinance as yf
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def test_setup():
    print("___starting____")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if(api_key):
        print("fetching done")
    else:
        print("issue")
    print("---testing connection---")
    try:
        llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
        response = llm.invoke("Say 'Hello")
        print(response.content)
    except Exception as e:
        print("issue with the model")
        
    print("Testing stock connection")
    try:
        apple = yf.Ticker("AAPL")
        price = apple.history(period = "1d")['Close'].iloc[0]
        print("Sucess", price)
    except Exception as e:
        print("error")
    
    print("check done")
    
if __name__ == "__main__":
    test_setup()