import streamlit as st
import pandas as pd
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ---------------------------------------------------------
# 🔧 PATH SETUP
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# ---------------------------------------------------------
# 📦 IMPORT BACKEND MODULES
# ---------------------------------------------------------
try:
    from rag.tools import DocumentTool
    from process_data import get_merged_data
    from model_training import predict_next_day
except ImportError:
    try:
        from src.rag.tools import DocumentTool
        from src.process_data import get_merged_data
        from src.model_training import predict_next_day
    except ImportError as e:
        st.error(f"❌ Critical Import Error: {e}")
        st.stop()

# ---------------------------------------------------------
# 🎨 PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="NEXUS AI | Finance Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    div[data-testid="stMetric"] { background-color: #1a1a2e; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    div[data-testid="stMetricLabel"] { color: #00ff88; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #ffffff; }
    h1, h2, h3 { color: #00ff88; font-family: 'monospace'; }
    .stButton>button { background-color: #00ff88; color: black; font-weight: bold; border-radius: 8px; border: none; }
    .stButton>button:hover { background-color: #00cc6a; color: white; }
    .stTextInput>div>div>input { background-color: #1a1a2e; color: white; border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🧠 SESSION STATE
# ---------------------------------------------------------
if 'doc_tool' not in st.session_state:
    st.session_state['doc_tool'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! I am Nexus AI. Ask me about stock predictions or upload a report to chat about fundamentals."}]
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

# ---------------------------------------------------------
# ⚙️ SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    st.markdown("### 1. Market Data")
    ticker = st.text_input("Stock Ticker", value="ITC.NS").upper()
    
    if st.button("🚀 Analyze Stock", use_container_width=True):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # 1. Prediction
                pred = predict_next_day(ticker)
                st.session_state['prediction'] = pred
                
                # 2. Historical Data
                df = get_merged_data(ticker, "2023-01-01", datetime.now().strftime('%Y-%m-%d'))
                
                # Normalize columns to lowercase to prevent KeyError
                df.columns = df.columns.str.lower()
                st.session_state['data'] = df
                
                st.success("Analysis Complete!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 2. Documents (RAG)")
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")
    
    if uploaded_file:
        if st.button("📂 Process & Index PDF"):
            with st.spinner("Indexing Document..."):
                try:
                    if not os.path.exists("documents"): os.makedirs("documents")
                    file_path = os.path.join("documents", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    tool = DocumentTool()
                    success = tool.ingest(file_path)
                    if success:
                        st.session_state['doc_tool'] = tool
                        st.success("✅ Document Indexed!")
                    else:
                        st.error("Failed to index.")
                except Exception as e:
                    st.error(f"RAG Error: {e}")

# ---------------------------------------------------------
# 📊 MAIN DASHBOARD
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📈 Dashboard & Charts", "💬 AI Chat Analyst"])

with tab1:
    st.title(f"📊 {ticker} Market Intelligence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = 0.0
    if st.session_state['data'] is not None:
        try:
            current_price = st.session_state['data']['close'].iloc[-1]
        except KeyError:
             st.error("Column 'close' not found.")
    
    pred_price = st.session_state['prediction']
    
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}" if current_price else "N/A")
    with col2:
        if pred_price:
            delta = ((pred_price - current_price) / current_price) * 100
            st.metric("Predicted (Tomorrow)", f"₹{pred_price:.2f}", f"{delta:.2f}%")
        else:
            st.metric("Predicted", "Pending...")
    with col3:
        signal = "BULLISH" if pred_price and pred_price > current_price else "BEARISH"
        st.metric("AI Signal", signal)
    with col4:
        st.metric("Model Confidence", "85%")

    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.subheader("Price History (Candlestick)")
        if st.session_state['data'] is not None:
            df = st.session_state['data']
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close'],
                name=ticker
            )])
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run analysis to see charts.")

    with col_chart2:
        st.subheader("Sentiment Trend")
        if st.session_state['data'] is not None:
            if 'sentiment_score' in st.session_state['data'].columns:
                sent_data = st.session_state['data']
            else:
                sent_data = st.session_state['data'].copy()
                sent_data['sentiment_score'] = 0 
            
            fig_sent = px.area(sent_data, x=sent_data.index, y='sentiment_score', line_shape='spline')
            # ✅ FIXED: Changed fill_color to fillcolor
            fig_sent.update_traces(line_color='#00ff88', fillcolor='rgba(0, 255, 136, 0.2)')
            fig_sent.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.info("Run analysis to see sentiment.")

with tab2:
    st.subheader("💬 Chat with Nexus AI")
    
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state['messages']:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
    
    if prompt := st.chat_input("Ask about price, risks, or the uploaded report..."):
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = ""
                
                keywords_market = ['price', 'buy', 'sell', 'target', 'prediction', 'forecast']
                is_market = any(w in prompt.lower() for w in keywords_market)
                
                if is_market:
                    pred = st.session_state.get('prediction', 'Unavailable')
                    response_text = f"**Technical Analysis:**\nMy LSTM model predicts a target of **₹{pred:.2f}** for tomorrow."
                    if pred == 'Unavailable':
                        response_text += " (Please click 'Analyze Stock' in the sidebar first.)"
                
                elif st.session_state['doc_tool']:
                    try:
                        res = st.session_state['doc_tool'].query(prompt)
                        response_text = f"**Fundamental Analysis:**\n{res['answer']}"
                    except Exception as e:
                        response_text = f"Error querying document: {e}"
                
                else:
                    response_text = "I can answer questions about the **Stock Price** (if you click Analyze) or the **Annual Report** (if you upload a PDF)."

                st.markdown(response_text)
                st.session_state['messages'].append({"role": "assistant", "content": response_text})