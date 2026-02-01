import argparse
import pandas as pd
import os
import sys

# ---------------------------------------------------------
# 🔧 PATH FIX: Ensure Python can find 'src' and current folder
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)  # Add src/ to path
sys.path.append(os.path.dirname(current_dir))  # Add parent folder to path

# 1. Import Stock Prediction Module (The Trader)
try:
    from process_data import get_merged_data
    from model_training import train_model, predict_next_day, SEQ_LENGTH
except ImportError:
    print("⚠️ Warning: Stock prediction modules (process_data.py) not found.")
    print("   (Agent will run in 'Document Only' mode if PDF is provided)")

# 2. Import RAG/Document Tool (The Analyst)
try:
    from rag.tools import DocumentTool
except ImportError as e:
    print(f"⚠️ Warning: RAG modules not found: {e}")
    DocumentTool = None

def run_prediction_task(ticker):
    """
    Runs the LSTM Stock Prediction pipeline.
    """
    # Check if module imports worked before running
    if 'get_merged_data' not in globals():
        print("❌ Error: Stock prediction modules are missing. Cannot predict price.")
        return None

    start_date = "2020-01-01" 
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    print(f"\n📊 Fetching market data for {ticker}...")
    try:
        df = get_merged_data(ticker, start_date, end_date)
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None
    
    if df is None or df.empty:
        print("❌ No data found.")
        return None

    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv(f"data/{ticker}_with_sentiment.csv", index=False)
    
    model_path = f"models/{ticker}_lstm.pth"
    if not os.path.exists("models"):
        os.makedirs("models")

    if len(df) > SEQ_LENGTH + 10:
        if not os.path.exists(model_path):
             print("🔄 Training new model...")
             train_model(ticker)
        else:
             print("✅ Using existing trained model.")
    
    try:
        price = predict_next_day(ticker)
        return price
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def run_pipeline(ticker, doc_path=None):
    print(f"\n🤖 FINANCIAL AGENT ACTIVATED: Analyzing {ticker}...")
    
    doc_tool = None
    
    # 1. Initialize RAG Tool
    if doc_path:
        if DocumentTool is None:
            print("❌ Error: Cannot load document because RAG module failed to import.")
            print("   (Did you create src/rag/tools.py?)")
        else:
            print(f"\n📂 Loading Report: {doc_path}")
            try:
                doc_tool = DocumentTool()
                success = doc_tool.ingest(doc_path)
                if success:
                    print("✅ RAG System Ready!")
                else:
                    print("⚠️ RAG Failed. Switching to Stock-Only mode.")
                    doc_tool = None
            except Exception as e:
                print(f"❌ RAG Initialization Error: {e}")
                doc_tool = None

    # --- INTERACTIVE MODE ---
    print("\n" + "="*50)
    print("💡 AGENT READY! Type 'exit' to quit.")
    print("👉 Ask about Price: 'Should I buy?', 'Prediction'")
    print("👉 Ask about PDF: 'What is the revenue?', 'Risks'")
    print("="*50)
        
    while True:
        try:
            user_query = input("\n❓ You: ")
            if user_query.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            if not user_query.strip():
                continue

            market_keywords = ['buy', 'sell', 'price', 'prediction', 'forecast', 'target', 'tomorrow', 'trend']
            is_market_query = any(word in user_query.lower() for word in market_keywords)

            # SCENARIO A: Market Question
            if is_market_query:
                print("\n⚙️  Running Technical Analysis...")
                pred_price = run_prediction_task(ticker)
                
                fund_insight = "Not available (No PDF loaded)"
                if doc_tool:
                    print("⚙️  Consulting Annual Report...")
                    rag_res = doc_tool.query("What is the future outlook and key risks? Summarize briefly.")
                    fund_insight = rag_res['answer']

                print("\n" + "="*45)
                print(f"🤖 AGENT REPORT: {ticker}")
                print("-" * 45)
                if pred_price:
                    print(f"📈 PRICE TARGET (Tomorrow):  ${pred_price:.2f}")
                else:
                    print("📈 PRICE TARGET: Unavailable")
                print("-" * 45)
                print(f"📜 FUNDAMENTAL INSIGHT:\n   {fund_insight}")
                print("="*45)

            # SCENARIO B: Document Question
            elif doc_tool:
                print("⏳ Thinking...")
                response = doc_tool.query(user_query)
                print(f"\n🧠 Answer: {response['answer']}")
                if 'source_text' in response and response['source_text']:
                    print(f"📄 Source: \"{response['source_text']}...\"")
            
            # SCENARIO C: Generic
            else:
                print("⚠️ I can only answer stock questions. (Load a PDF to ask about documents)")
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='ITC.NS', help='Stock Ticker')
    parser.add_argument('--doc', type=str, help='Path to PDF Report')
    
    args = parser.parse_args()
    run_pipeline(args.ticker, args.doc)