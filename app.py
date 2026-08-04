import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from stocknews import StockNews
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import warnings
warnings.filterwarnings("ignore")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(
    page_title="StockLab",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
except Exception:
    GEMINI_API_KEY = ""
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anybody:wght@300;400;500;600;700;900&family=Geist+Mono:wght@300;400;500;600&family=Geist:wght@300;400;500;600&display=swap');

:root {
  --bg0:        #0f0e0d;
  --bg1:        #161412;
  --bg2:        #1e1b18;
  --bg3:        #262219;
  --border:     rgba(255,255,255,0.06);
  --border-hi:  rgba(255,255,255,0.12);
  --gold:       #d4a853;
  --gold-dim:   #9a7535;
  --gold-glow:  rgba(212,168,83,0.15);
  --green:      #4ade80;
  --green-dim:  rgba(74,222,128,0.12);
  --red:        #f87171;
  --red-dim:    rgba(248,113,113,0.12);
  --amber:      #fbbf24;
  --text0:      #f5f0e8;
  --text1:      #a89880;
  --text2:      #6b5e4e;
  --mono:       'Geist Mono', monospace;
  --display:    'Anybody', sans-serif;
  --body:       'Geist', sans-serif;
  --r:          6px;
  --r-lg:       10px;
}

html, body, .stApp {
  background: var(--bg0) !important;
  color: var(--text0);
  font-family: var(--body);
}

.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 9999;
  opacity: .018;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
  background-size: 200px 200px;
}

.stApp::after {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background:
    radial-gradient(ellipse 700px 400px at 20% -10%,  rgba(212,168,83,0.05) 0%, transparent 70%),
    radial-gradient(ellipse 500px 500px at 90% 110%,  rgba(212,168,83,0.03) 0%, transparent 70%);
}

.block-container {
  position: relative;
  z-index: 1;
  padding: 0 !important;
  max-width: 100% !important;
}
section[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border) !important;
  width: 260px !important;
  min-width: 260px !important;
}
section[data-testid="stSidebar"] > div:first-child {
  padding: 28px 20px 20px !important;
  width: 100% !important;
  position: relative !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding: 28px 20px 20px !important;
  overflow-y: auto !important;
}
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] div.stButton {
  max-width: 100% !important;
  width: 100% !important;
}

.sl-wordmark {
  font-family: var(--display);
  font-size: 22px;
  font-weight: 900;
  letter-spacing: -0.5px;
  color: var(--text0);
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 2px;
}
.sl-wordmark-dot {
  width: 8px; height: 8px;
  background: var(--gold);
  border-radius: 50%;
  box-shadow: 0 0 10px var(--gold);
  animation: breathe 3s ease-in-out infinite;
}
@keyframes breathe {
  0%,100% { box-shadow: 0 0 8px var(--gold); }
  50%      { box-shadow: 0 0 18px var(--gold), 0 0 32px rgba(212,168,83,0.3); }
}
.sl-sub {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text2);
  margin-bottom: 28px;
}

.field-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--text2);
  margin-bottom: 5px;
  margin-top: 16px;
}

.sl-rule {
  height: 1px;
  background: var(--border);
  margin: 20px 0;
}
.sl-rule-gold {
  height: 1px;
  background: linear-gradient(90deg, var(--gold-dim), transparent);
  margin: 20px 0;
}

.stTextInput input {
  background: var(--bg2) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: var(--r) !important;
  color: var(--text0) !important;
  font-family: var(--mono) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  letter-spacing: 0.5px;
  transition: border-color .15s, box-shadow .15s;
}
.stTextInput input:focus {
  border-color: var(--gold-dim) !important;
  box-shadow: 0 0 0 3px rgba(212,168,83,0.1) !important;
  outline: none !important;
}
.stTextInput label { display: none !important; }

.stSlider label { display: none !important; }
.stSlider [role="slider"] {
  background: var(--gold) !important;
  border: 2px solid var(--bg0) !important;
  width: 14px !important; height: 14px !important;
  box-shadow: 0 0 8px rgba(212,168,83,0.5) !important;
}
.stSlider [data-testid="stSliderTrackActive"] {
  background: var(--gold-dim) !important;
}
.stSlider [data-testid="stSlider"] > div > div > div:first-child {
  background: var(--bg3) !important;
}

div.stButton > button {
  width: 100% !important;
  background: var(--gold) !important;
  color: #0f0e0d !important;
  border: none !important;
  border-radius: var(--r) !important;
  font-family: var(--display) !important;
  font-weight: 700 !important;
  font-size: 11px !important;
  letter-spacing: 2.5px !important;
  text-transform: uppercase !important;
  padding: 11px 0 !important;
  cursor: pointer !important;
  transition: opacity .15s, transform .1s, box-shadow .2s !important;
  box-shadow: 0 2px 16px rgba(212,168,83,0.25) !important;
  margin-top: 4px !important;
}
div.stButton > button:hover {
  opacity: .92 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 24px rgba(212,168,83,0.35) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

div[data-testid="stMetric"] {
  background: var(--bg1) !important;
  border: 1px solid var(--border) !important;
  border-top: 2px solid var(--gold-dim) !important;
  border-radius: var(--r-lg) !important;
  padding: 16px 20px !important;
  position: relative;
  overflow: hidden;
  transition: border-color .2s, box-shadow .2s;
}
div[data-testid="stMetric"]:hover {
  border-color: var(--border-hi) !important;
  border-top-color: var(--gold) !important;
  box-shadow: 0 0 30px rgba(212,168,83,0.08) !important;
}
div[data-testid="stMetricLabel"] p {
  font-family: var(--mono) !important;
  font-size: 9px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  color: var(--text2) !important;
}
div[data-testid="stMetricValue"] {
  font-family: var(--display) !important;
  font-size: 26px !important;
  font-weight: 700 !important;
  color: var(--text0) !important;
  letter-spacing: -0.5px !important;
}
div[data-testid="stMetricDelta"] {
  font-family: var(--mono) !important;
  font-size: 11px !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0 !important;
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 !important;
  margin-bottom: 20px !important;
  width: 100% !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  color: var(--text2) !important;
  padding: 10px 20px !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  background: transparent !important;
  transition: color .15s, border-color .15s !important;
  height: auto !important;
}
.stTabs [aria-selected="true"] {
  color: var(--gold) !important;
  border-bottom-color: var(--gold) !important;
  background: transparent !important;
}
div[data-baseweb="tab-highlight"] { display: none !important; }
div[data-baseweb="tab-border"]    { display: none !important; }

.stDataFrame {
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  overflow: hidden;
}
.stDataFrame th {
  background: var(--bg2) !important;
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  color: var(--gold-dim) !important;
  border-bottom: 1px solid var(--border) !important;
}
.stDataFrame td {
  font-family: var(--mono) !important;
  font-size: 12px !important;
}

.stChatMessage {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  padding: 14px 18px !important;
  margin-bottom: 8px !important;
}
[data-testid="stChatMessageContent"] p {
  font-family: var(--body) !important;
  font-size: 13.5px !important;
  line-height: 1.65 !important;
  color: var(--text0) !important;
}
.stChatInputContainer {
  background: var(--bg2) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: var(--r) !important;
}
.stChatInputContainer:focus-within {
  border-color: var(--gold-dim) !important;
  box-shadow: 0 0 0 3px rgba(212,168,83,0.08) !important;
}
.stChatInputContainer textarea {
  font-family: var(--body) !important;
  font-size: 13px !important;
  color: var(--text0) !important;
  background: transparent !important;
}

.stSpinner > div { border-top-color: var(--gold) !important; }

.stAlert { border-radius: var(--r) !important; font-family: var(--body) !important; }

.sec-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}
.sec-head-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text2);
}
.sec-head-line {
  flex: 1;
  height: 1px;
  background: var(--border);
}
.sec-head-dot {
  width: 4px; height: 4px;
  border-radius: 50%;
  background: var(--gold-dim);
}

.co-header {
  padding: 28px 32px 24px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 20px;
  background: var(--bg1);
  position: relative;
  overflow: hidden;
}
.co-header::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, var(--gold-dim) 0%, transparent 60%);
}
.co-name {
  font-family: var(--display);
  font-size: 32px;
  font-weight: 900;
  color: var(--text0);
  letter-spacing: -1px;
  line-height: 1;
}
.co-ticker {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--gold);
  background: rgba(212,168,83,0.08);
  border: 1px solid rgba(212,168,83,0.2);
  padding: 4px 10px;
  border-radius: 4px;
  letter-spacing: 1px;
}
.co-badge {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--text2);
  background: var(--bg2);
  border: 1px solid var(--border);
  padding: 3px 9px;
  border-radius: 4px;
}
.co-date {
  margin-left: auto;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text2);
}

.kpi-strip {
  padding: 20px 32px;
  border-bottom: 1px solid var(--border);
  background: var(--bg0);
}

.main-canvas {
  padding: 24px 32px 32px;
}

.nc {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-left: 2px solid var(--nc-color, var(--text2));
  border-radius: var(--r);
  padding: 12px 14px;
  margin-bottom: 8px;
  transition: border-color .15s, transform .15s;
  cursor: default;
}
.nc:hover {
  border-color: var(--border-hi);
  border-left-color: var(--nc-color, var(--text2));
  transform: translateX(2px);
}
.nc-title {
  font-family: var(--body);
  font-size: 12.5px;
  font-weight: 500;
  color: var(--text0);
  line-height: 1.45;
  margin-bottom: 6px;
}
.nc-meta {
  font-family: var(--mono);
  font-size: 9.5px;
  color: var(--text2);
  display: flex;
  gap: 10px;
  align-items: center;
}
.nc-badge {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  padding: 2px 7px;
  border-radius: 3px;
}

.sent-card {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 20px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.sent-value {
  font-family: var(--display);
  font-size: 28px;
  font-weight: 900;
  letter-spacing: -0.5px;
}
.sent-sub {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--text2);
  margin-top: 3px;
}
.sent-score-box {
  text-align: right;
}
.sent-score-val {
  font-family: var(--mono);
  font-size: 22px;
  font-weight: 600;
}
.sent-score-lbl {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--text2);
}

.idle-screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 80vh;
  gap: 16px;
  text-align: center;
}
.idle-logo {
  font-family: var(--display);
  font-size: 72px;
  font-weight: 900;
  letter-spacing: -3px;
  color: var(--text0);
  line-height: 1;
}
.idle-logo span { color: var(--gold); }
.idle-hint {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: var(--text2);
}
.idle-steps {
  display: flex;
  gap: 32px;
  margin-top: 8px;
}
.idle-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}
.idle-step-num {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--gold-dim);
  background: rgba(212,168,83,0.06);
  border: 1px solid rgba(212,168,83,0.12);
  width: 24px; height: 24px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
}
.idle-step-lbl {
  font-family: var(--body);
  font-size: 12px;
  color: var(--text2);
}

/* FIX: Keep the native toggle button visible by making the header transparent instead of display: none */
#MainMenu, footer, .stDeployButton { display: none !important; }
header { background: transparent !important; box-shadow: none !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg0); }
::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold-dim); }

hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
.anim { animation: fadeUp .35s ease both; }
.anim-d1 { animation-delay: .05s; }
.anim-d2 { animation-delay: .1s; }
.anim-d3 { animation-delay: .15s; }
.anim-d4 { animation-delay: .2s; }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
    <div class="sl-wordmark">
      <div class="sl-wordmark-dot"></div>
      StockLab
    </div>
    <div class="sl-sub">Market Intelligence</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sl-rule-gold"></div>', unsafe_allow_html=True)

    st.markdown('<div class="field-label">Asset Ticker</div>', unsafe_allow_html=True)
    user_input = st.text_input(
        "_ticker",
        value="TSLA",
        label_visibility="collapsed"
    ).upper()

    st.markdown('<div class="field-label">Historical Window</div>', unsafe_allow_html=True)
    years = st.slider("_years", 1, 10, 5, label_visibility="collapsed")

    st.markdown(
        f'<div style="font-family:var(--mono);font-size:10px;color:var(--gold-dim);margin-top:-8px;margin-bottom:4px;">'
        f'{years} yr{"s" if years>1 else ""} · ~{years*252} sessions'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="field-label">Forecast Horizon</div>', unsafe_allow_html=True)
    prediction_days = st.slider("_fdays", 1, 7, 1, label_visibility="collapsed")

    st.markdown(
        f'<div style="font-family:var(--mono);font-size:10px;color:var(--gold-dim);margin-top:-8px;margin-bottom:4px;">'
        f'{prediction_days} trading day{"s" if prediction_days>1 else ""} ahead'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sl-rule"></div>', unsafe_allow_html=True)

    run = st.button("RUN ANALYSIS", type="primary")

    if run:
        st.session_state['run_analysis'] = True
        st.session_state['chat_history'] = []
        st.session_state['vs'] = None

    if 'run_analysis' not in st.session_state:
        st.session_state['run_analysis'] = True

    st.markdown(f"""
    <div class="sl-rule"></div>
    <div style="font-family:var(--mono);font-size:9px;color:var(--text2);line-height:2;">
      <div>ENGINE · RandomForest v2</div>
      <div>EMBED · gemini-embedding-001</div>
      <div>LLM · gemini-2.5-flash</div>
      <div>NLP · VADER Sentiment</div>
      <div style="margin-top:8px;color:#2a221a;">
        build {date.today().strftime("%Y%m%d")}
      </div>
    </div>
    """, unsafe_allow_html=True)

def get_currency_symbol(code):
    return {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥"}.get(code, code + " ")

def search_global_market(query):
    try:
        r = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 1, "newsCount": 0},
            headers={'User-Agent': 'Mozilla/5.0'}, timeout=5
        )
        data = r.json()
        if data.get('quotes'):
            b = data['quotes'][0]
            return b['symbol'], b.get('longname', query)
    except:
        pass
    return query, query

@st.cache_data
def get_stock_data(user_query, years):
    start = (date.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    end   = date.today().strftime("%Y-%m-%d")
    
    ticker_found, company_name = search_global_market(user_query)
    currency_code = "USD"
    
    if not ticker_found:
        return None, None, None, None

    if ticker_found.endswith(".NS") or ticker_found.endswith(".BO"):
        currency_code = "INR"
    else:
        try:
            currency_code = yf.Ticker(ticker_found).info.get('currency', 'USD')
        except:
            pass

    raw = yf.Ticker(ticker_found).history(start=start, end=end)
    
    if raw.empty:
        return None, None, None, None
        
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
        
    raw.reset_index(inplace=True)
    raw['Date'] = pd.to_datetime(raw['Date']).dt.date
    raw.set_index('Date', inplace=True)
    
    return raw, currency_code, ticker_found, company_name

def calculate_indicators(df, pred_days):
    d = df.copy()
    delta = d['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain / loss))
    e12 = d['Close'].ewm(span=12, adjust=False).mean()
    e26 = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD']        = e12 - e26
    d['Signal_Line'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['SMA_20']      = d['Close'].rolling(20).mean()
    d['Std_Dev']     = d['Close'].rolling(20).std()
    d['Upper_Band']  = d['SMA_20'] + d['Std_Dev'] * 2
    d['Lower_Band']  = d['SMA_20'] - d['Std_Dev'] * 2
    d['OBV']         = (np.sign(d['Close'].diff()) * d['Volume']).fillna(0).cumsum()
    d['Target_Return'] = d['Close'].pct_change().shift(-pred_days)
    d.dropna(inplace=True)
    return d

@st.cache_resource(show_spinner=False)
def build_vs(ticker, _df_news, key):
    os.environ["GOOGLE_API_KEY"] = key
    t = yf.Ticker(ticker)
    summary = t.info.get('longBusinessSummary', 'No summary available.')
    docs = [Document(page_content=f"Corporate Profile — {ticker}: {summary}")]
    if _df_news is not None and not _df_news.empty:
        for i in range(len(_df_news)):
            r = _df_news.iloc[i]
            docs.append(Document(page_content=f"{r['published']} | {r['title']} | {r.get('summary','')}"))
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    
    return FAISS.from_documents(splits, GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"))

def rag_query(question, vs, key):
    os.environ["GOOGLE_API_KEY"] = key
    docs = vs.similarity_search(question, k=3)
    prompt = PromptTemplate(
        template="""You are an expert institutional financial analyst. Answer ONLY from the context below.
State clearly if something is not in the context. Never fabricate data.

Context:
{context}

Question: {question}

Analyst Response:""",
        input_variables=["context", "question"]
    )
    
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2) | StrOutputParser()
    return chain.invoke({"context": "\n\n".join(d.page_content for d in docs), "question": question})

def chart_layout(h=420):
    return dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=12, b=0),
        height=h,
        hovermode="x unified",
        font=dict(family='Geist Mono, monospace', size=10, color='#6b5e4e'),
        xaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(size=9, color='#4a3c2e')),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)',
                   zeroline=False, showline=False,
                   tickfont=dict(size=9, color='#4a3c2e')),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=9)),
        hoverlabel=dict(bgcolor='#1e1b18', bordercolor='rgba(212,168,83,0.2)',
                        font=dict(family='Geist Mono, monospace', size=11)),
    )

if not st.session_state.get('run_analysis', False):
    st.markdown("""
    <div class="idle-screen">
      <div class="idle-logo">Stock<span>Lab</span></div>
      <div class="idle-hint">Institutional Market Intelligence Platform</div>
      <div style="height:8px;"></div>
      <div class="idle-steps">
        <div class="idle-step">
          <div class="idle-step-num">1</div>
          <div class="idle-step-lbl">Enter ticker</div>
        </div>
        <div class="idle-step">
          <div class="idle-step-num">2</div>
          <div class="idle-step-lbl">Set horizon</div>
        </div>
        <div class="idle-step">
          <div class="idle-step-num">3</div>
          <div class="idle-step-lbl">Run analysis</div>
        </div>
      </div>
      <div style="margin-top:24px;font-family:var(--mono);font-size:9px;color:#2a221a;letter-spacing:2px;">
        FAISS · VADER · RandomForest · Gemini 2.5
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


with st.spinner(f"Loading {user_input}…"):
    df_raw, currency_code, resolved_ticker, company_name = get_stock_data(user_input, years)

if df_raw is None or df_raw.empty:
    st.error(f"No data found for **{user_input}**. Try AAPL, TSLA, RELIANCE, INFY.")
    st.stop()

csym = get_currency_symbol(currency_code)
df_q = calculate_indicators(df_raw, prediction_days)
feats = ['RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band']
X, y = df_q[feats], df_q['Target_Return']
split = int(len(X) * 0.85)
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X.iloc[:split], y.iloc[:split])

cf          = X.iloc[[-1]]
pred_ret    = rf.predict(cf)[0]
cur_price   = df_q['Close'].iloc[-1]
tgt_price   = cur_price * (1 + pred_ret)
price_delta = tgt_price - cur_price
rsi_val     = cf['RSI'].values[0]
rsi_sig     = "OVERBOUGHT" if rsi_val > 70 else "OVERSOLD" if rsi_val < 30 else "NEUTRAL"
vol_today   = int(df_q['Volume'].iloc[-1])
day_range   = f"{csym}{df_q['Low'].iloc[-1]:,.2f} – {csym}{df_q['High'].iloc[-1]:,.2f}"

df_52 = df_q.iloc[-252:] if len(df_q) >= 252 else df_q
hi52  = df_52['Close'].max()
lo52  = df_52['Close'].min()

try:
    sn     = StockNews([resolved_ticker], save_news=False)
    df_news = sn.read_rss()
except:
    df_news = None

rsi_color = "#f87171" if rsi_val > 70 else "#4ade80" if rsi_val < 30 else "#d4a853"
forecast_color = "#4ade80" if pred_ret > 0 else "#f87171"
arrow = "▲" if pred_ret > 0 else "▼"

st.markdown(f"""
<div class="co-header anim">
  <div>
    <div class="co-name">{company_name or user_input}</div>
    <div style="margin-top:4px;font-family:var(--mono);font-size:11px;color:var(--text2);">
      {date.today().strftime("%A, %d %B %Y")}
    </div>
  </div>
  <div class="co-ticker">{resolved_ticker}</div>
  <div class="co-badge">{currency_code}</div>
  <div class="co-badge">{years}yr history</div>
  <div style="margin-left:auto;display:flex;gap:32px;text-align:right;">
    <div>
      <div style="font-family:var(--mono);font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--text2);">52W Range</div>
      <div style="font-family:var(--mono);font-size:12px;color:var(--text0);margin-top:3px;">{csym}{lo52:,.2f} – {csym}{hi52:,.2f}</div>
    </div>
    <div>
      <div style="font-family:var(--mono);font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--text2);">Volume</div>
      <div style="font-family:var(--mono);font-size:12px;color:var(--text0);margin-top:3px;">{vol_today/1e6:.2f}M</div>
    </div>
    <div>
      <div style="font-family:var(--mono);font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--text2);">RSI</div>
      <div style="font-family:var(--mono);font-size:12px;color:{rsi_color};margin-top:3px;">{rsi_val:.1f}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


st.markdown('<div style="padding:20px 32px 0;background:var(--bg0);">', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Last Price",        f"{csym}{cur_price:,.2f}")
k2.metric(f"Forecast ({prediction_days}d)", f"{arrow} {abs(pred_ret)*100:.2f}%")
k3.metric("Target Price",      f"{csym}{tgt_price:,.2f}", delta=f"{price_delta:+,.2f}")
k4.metric("RSI Signal",        rsi_sig)
k5.metric("Day Range",         day_range)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:1px;background:var(--border);margin:20px 32px 0;"></div>', unsafe_allow_html=True)

st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["Price & Volatility", "Momentum", "Raw Data", "AI Analyst"])


with tab1:
    col_chart, col_news = st.columns([2.4, 1], gap="large")

    with col_chart:
        st.markdown('<div class="sec-head"><div class="sec-head-dot"></div><div class="sec-head-label">Price with Bollinger Bands</div><div class="sec-head-line"></div></div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_q.index, y=df_q['Upper_Band'], mode='lines', name='Upper BB',
            line=dict(color='rgba(212,168,83,0.25)', width=1, dash='dot'),
        ))
        fig.add_trace(go.Scatter(
            x=df_q.index, y=df_q['Lower_Band'], mode='lines', name='Lower BB',
            line=dict(color='rgba(212,168,83,0.25)', width=1, dash='dot'),
            fill='tonexty', fillcolor='rgba(212,168,83,0.018)',
        ))
        fig.add_trace(go.Scatter(
            x=df_q.index, y=df_q['SMA_20'], mode='lines', name='SMA 20',
            line=dict(color='rgba(212,168,83,0.45)', width=1.2, dash='dot'),
        ))
        fig.add_trace(go.Scatter(
            x=df_q.index, y=df_q['Close'], mode='lines', name='Close',
            line=dict(color='#d4a853', width=2),
            fill='tozeroy', fillcolor='rgba(212,168,83,0.03)',
        ))
        layout = chart_layout(460)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_news:
        st.markdown('<div class="sec-head"><div class="sec-head-dot"></div><div class="sec-head-label">News Feed · NLP Scored</div><div class="sec-head-line"></div></div>', unsafe_allow_html=True)

        news_box = st.container(height=470)
        with news_box:
            if df_news is None or df_news.empty:
                st.info("No recent news found for this ticker.")
            else:
                sia = SentimentIntensityAnalyzer()
                scores = []
                for i in range(min(12, len(df_news))):
                    row   = df_news.iloc[i]
                    score = sia.polarity_scores(row['title'])['compound']
                    scores.append(score)
                    if score > 0.1:
                        c, badge_cls, badge_lbl = "#4ade80", "background:rgba(74,222,128,0.1);color:#4ade80;border:1px solid rgba(74,222,128,0.2);", "BULL"
                    elif score < -0.1:
                        c, badge_cls, badge_lbl = "#f87171", "background:rgba(248,113,113,0.1);color:#f87171;border:1px solid rgba(248,113,113,0.2);", "BEAR"
                    else:
                        c, badge_cls, badge_lbl = "#d4a853", "background:rgba(212,168,83,0.08);color:#d4a853;border:1px solid rgba(212,168,83,0.15);", "NEUT"

                    pub = str(row.get('published', ''))[:16]
                    st.markdown(f"""
                    <div class="nc" style="--nc-color:{c};">
                      <div class="nc-title">{row['title']}</div>
                      <div class="nc-meta">
                        <span>{pub}</span>
                        <span class="nc-badge" style="{badge_cls}">{badge_lbl} {score:+.2f}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="sec-head"><div class="sec-head-dot"></div><div class="sec-head-label">MACD · RSI Oscillators</div><div class="sec-head-line"></div></div>', unsafe_allow_html=True)

    fig2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.55, 0.45],
    )
    macd_diff = df_q['MACD'] - df_q['Signal_Line']
    colors = ['rgba(212,168,83,0.55)' if v >= 0 else 'rgba(248,113,113,0.45)' for v in macd_diff]
    fig2.add_trace(go.Bar(x=df_q.index, y=macd_diff, marker_color=colors, name='Divergence', showlegend=False), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df_q.index, y=df_q['MACD'], mode='lines', name='MACD', line=dict(color='#d4a853', width=1.8)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df_q.index, y=df_q['Signal_Line'], mode='lines', name='Signal', line=dict(color='rgba(212,168,83,0.4)', width=1.2, dash='dot')), row=1, col=1)

    rsi_colors = ['#f87171' if v > 70 else '#4ade80' if v < 30 else '#d4a853' for v in df_q['RSI']]
    fig2.add_trace(go.Scatter(x=df_q.index, y=df_q['RSI'], mode='lines', name='RSI',
                              line=dict(color='#d4a853', width=1.8),
                              fill='tozeroy', fillcolor='rgba(212,168,83,0.03)'), row=2, col=1)
    fig2.add_hline(y=70, line_dash="dot", line_color="rgba(248,113,113,0.4)", row=2, col=1)
    fig2.add_hline(y=30, line_dash="dot", line_color="rgba(74,222,128,0.4)", row=2, col=1)
    fig2.add_hrect(y0=70, y1=100, fillcolor="rgba(248,113,113,0.025)", line_width=0, row=2, col=1)
    fig2.add_hrect(y0=0,  y1=30,  fillcolor="rgba(74,222,128,0.025)",  line_width=0, row=2, col=1)

    layout2 = chart_layout(520)
    layout2['xaxis2'] = dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=9, color='#4a3c2e'))
    layout2['yaxis2'] = dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False, range=[0,100], tickfont=dict(size=9, color='#4a3c2e'))
    fig2.update_layout(**layout2)
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})


with tab3:
    st.markdown('<div class="sec-head"><div class="sec-head-dot"></div><div class="sec-head-label">Last 60 Sessions · Computed Features</div><div class="sec-head-line"></div></div>', unsafe_allow_html=True)

    show_cols = ['Open','High','Low','Close','Volume','RSI','MACD','Signal_Line','Upper_Band','Lower_Band']
    avail     = [c for c in show_cols if c in df_q.columns]
    fmt       = {c: "{:,.2f}" for c in avail if c != 'Volume'}
    fmt['Volume'] = "{:,.0f}"
    st.dataframe(
        df_q[avail].tail(60).style.format(fmt),
        use_container_width=True, height=500
    )


with tab4:
    left_ai, right_ai = st.columns([1, 1.4], gap="large")

    with left_ai:
        st.markdown('<div class="sec-head"><div class="sec-head-dot"></div><div class="sec-head-label">Sentiment Overview</div><div class="sec-head-line"></div></div>', unsafe_allow_html=True)

        if df_news is not None and not df_news.empty:
            sia    = SentimentIntensityAnalyzer()
            scores = [sia.polarity_scores(r['title'])['compound'] for _, r in df_news.iterrows()]
            avg    = float(np.mean(scores))

            if avg > 0.15:
                s_lbl, s_col = "BULLISH", "#4ade80"
            elif avg < -0.15:
                s_lbl, s_col = "BEARISH", "#f87171"
            else:
                s_lbl, s_col = "NEUTRAL", "#d4a853"

            st.markdown(f"""
            <div class="sent-card">
              <div>
                <div class="sent-value" style="color:{s_col};">{s_lbl}</div>
                <div class="sent-sub">Aggregate NLP Signal</div>
              </div>
              <div class="sent-score-box">
                <div class="sent-score-val" style="color:{s_col};">{avg:+.3f}</div>
                <div class="sent-score-lbl">VADER Score</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            fig_sent = go.Figure()
            bar_colors = ['rgba(74,222,128,0.7)' if s > 0.1 else 'rgba(248,113,113,0.7)' if s < -0.1 else 'rgba(212,168,83,0.5)' for s in scores[:10]]
            fig_sent.add_trace(go.Bar(
                x=list(range(1, len(scores[:10])+1)),
                y=scores[:10],
                marker_color=bar_colors,
                marker_line_width=0,
                name='Article Score'
            ))
            fig_sent.add_hline(y=avg, line_dash="dot", line_color="rgba(212,168,83,0.6)", line_width=1)
            l = chart_layout(200)
            l['xaxis']['title'] = 'Article'
            l['margin'] = dict(l=0,r=0,t=4,b=0)
            fig_sent.update_layout(**l)
            st.plotly_chart(fig_sent, use_container_width=True, config={'displayModeBar': False})

            bull = sum(1 for s in scores if s > 0.1)
            bear = sum(1 for s in scores if s < -0.1)
            neut = len(scores) - bull - bear

            st.markdown(f"""
            <div style="display:flex;gap:8px;margin-top:8px;">
              <div style="flex:1;background:rgba(74,222,128,0.06);border:1px solid rgba(74,222,128,0.15);border-radius:6px;padding:12px;text-align:center;">
                <div style="font-family:var(--display);font-size:22px;font-weight:700;color:#4ade80;">{bull}</div>
                <div style="font-family:var(--mono);font-size:9px;letter-spacing:1.5px;color:var(--text2);margin-top:2px;">BULLISH</div>
              </div>
              <div style="flex:1;background:rgba(212,168,83,0.06);border:1px solid rgba(212,168,83,0.15);border-radius:6px;padding:12px;text-align:center;">
                <div style="font-family:var(--display);font-size:22px;font-weight:700;color:#d4a853;">{neut}</div>
                <div style="font-family:var(--mono);font-size:9px;letter-spacing:1.5px;color:var(--text2);margin-top:2px;">NEUTRAL</div>
              </div>
              <div style="flex:1;background:rgba(248,113,113,0.06);border:1px solid rgba(248,113,113,0.15);border-radius:6px;padding:12px;text-align:center;">
                <div style="font-family:var(--display);font-size:22px;font-weight:700;color:#f87171;">{bear}</div>
                <div style="font-family:var(--mono);font-size:9px;letter-spacing:1.5px;color:var(--text2);margin-top:2px;">BEARISH</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No news articles found for this ticker.")

    with right_ai:
        st.markdown('<div class="sec-head"><div class="sec-head-dot"></div><div class="sec-head-label">RAG Document Analyst · Gemini 2.5</div><div class="sec-head-line"></div></div>', unsafe_allow_html=True)

        if not GEMINI_API_KEY:
            st.warning("Set `GOOGLE_API_KEY` in `.streamlit/secrets.toml` or as an environment variable to activate the AI analyst.")
        else:
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []
            if 'vs' not in st.session_state or st.session_state.get('vs') is None:
                with st.spinner("Indexing corporate documents…"):
                    try:
                        st.session_state['vs'] = build_vs(resolved_ticker, df_news, GEMINI_API_KEY)
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                            st.error("⚠️ **API Rate Limit Exceeded:** You made too many requests. Wait ~60 seconds and run the analysis again.")
                        else:
                            st.error(f"Indexing error: {error_msg}")
                        st.stop()

            chat_win = st.container(height=460)
            with chat_win:
                if not st.session_state['chat_history']:
                    st.markdown(f"""
                    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                                height:100%;padding:40px;text-align:center;gap:10px;">
                      <div style="font-family:var(--display);font-size:40px;font-weight:900;color:var(--border-hi);">◈</div>
                      <div style="font-family:var(--body);font-size:13px;color:var(--text2);">
                        Ask about <strong style="color:var(--text1);">{company_name or resolved_ticker}</strong>
                      </div>
                      <div style="font-family:var(--mono);font-size:10px;color:var(--text2);line-height:2;">
                        business model · revenue streams · key risks · recent news
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                for m in st.session_state['chat_history']:
                    with st.chat_message(m['role']):
                        st.markdown(m['content'])

            if prompt := st.chat_input("Ask about the company, risks, or recent news…"):
                st.session_state['chat_history'].append({"role": "user", "content": prompt})
                with chat_win:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Querying knowledge base…"):
                            try:
                                ans = rag_query(prompt, st.session_state['vs'], GEMINI_API_KEY)
                                st.markdown(ans)
                                st.session_state['chat_history'].append({"role": "assistant", "content": ans})
                            except Exception as e:
                                error_msg = str(e)
                                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                                    st.error("⚠️ **API Limit Exceeded:** Please wait 60 seconds.")
                                else:
                                    st.error(f"Query error: {error_msg}")

st.markdown('</div>', unsafe_allow_html=True)