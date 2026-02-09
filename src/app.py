import streamlit as st
import pandas as pd
import os
import sys

# Add src directory to path so imports work from project root
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)

# Add both directories to sys.path for flexible imports
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change working directory to project root for relative paths (models/, data/, etc.)
os.chdir(project_root)

# Import your existing backend logic
try:
    from rag.tools import DocumentTool
    from process_data import get_merged_data
    from model_training import predict_next_day
except ImportError as e:
    st.error(f"❌ Critical Import Error: {e}. Please ensure 'rag', 'process_data.py', and 'model_training.py' are in the 'src' folder.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Financial AI Agent", layout="wide")

st.title("🤖 Hybrid Financial AI Agent")
st.markdown("### Technical Analysis (LSTM) + Fundamental Analysis (Gemini RAG)")

# --- SIDEBAR: Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Stock Ticker (Yahoo Finance)", value="ITC.NS")
    
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")
    
    if st.button("🔄 Initialize Agent"):
        with st.spinner("Processing Data..."):
            # 1. Save Uploaded PDF
            pdf_path = None
            if uploaded_file:
                if not os.path.exists("documents"):
                    os.makedirs("documents")
                pdf_path = os.path.join("documents", uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize RAG
                st.session_state['doc_tool'] = DocumentTool()
                st.session_state['doc_tool'].ingest(pdf_path)
                st.success("✅ PDF Loaded & Vectorized!")
            
            # 2. Run Stock Prediction
            try:
                csv_path = f"data/{ticker}_with_sentiment.csv"
                model_path = f"models/{ticker}_lstm.pth"
                
                # Check if this is a new ticker
                is_new_ticker = not os.path.exists(csv_path) or not os.path.exists(model_path)
                
                if is_new_ticker:
                    st.warning(f"⚠️ New ticker detected: {ticker}")
                    st.info("🔄 Step 1: Fetching stock data and news (this may take 2-3 minutes)...")
                
                # predict_next_day now handles auto-training
                prediction = predict_next_day(ticker)
                
                if prediction is None:
                    st.error(f"❌ Could not generate prediction for {ticker}. Please check if the ticker symbol is valid.")
                else:
                    st.session_state['price_pred'] = prediction
                    
                    if is_new_ticker:
                        st.success(f"✅ Model trained and ready!")
                
                # Load chart data
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    st.session_state['chart_data'] = df
                    st.success(f"✅ Prediction for {ticker} Complete!")
                else:
                    st.error(f"❌ Data file not found for {ticker}.")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- MAIN PANEL ---
col1, col2 = st.columns([1, 1])

# LEFT COLUMN: Stock Data
with col1:
    st.subheader(f"📈 {ticker} Market Outlook")
    
    if 'price_pred' in st.session_state and st.session_state['price_pred'] is not None:
        pred = st.session_state['price_pred']
        st.metric(label="Predicted Closing Price (Tomorrow)", value=f"₹{pred:.2f}")
    
    if 'chart_data' in st.session_state:
        st.line_chart(st.session_state['chart_data']['close'])
    else:
        st.info("Click 'Initialize Agent' to see market data.")

# RIGHT COLUMN: AI Chat
with col2:
    st.subheader("💬 AI Analyst (RAG)")
    
    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask about the report or the stock..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            response_text = "⚠️ Please upload a PDF and Initialize the Agent first."
            
            if 'doc_tool' in st.session_state:
                with st.spinner("Thinking..."):
                    # Check if asking about stock or pdf
                    if any(w in prompt.lower() for w in ['price', 'buy', 'sell', 'target']):
                         pred = st.session_state.get('price_pred', 'Unavailable')
                         response_text = f"Based on my LSTM Technical Analysis, the predicted price target for tomorrow is **₹{pred:.2f}**."
                    else:
                        # RAG Query
                        res = st.session_state['doc_tool'].query(prompt)
                        response_text = res['answer']
                        if res.get('source_text'):
                             response_text += f"\n\n> *Source: {res['source_text']}...*"
            
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})