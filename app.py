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

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(page_title="StockLab", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    .sidebar-logo {
        font-family: 'Inter', sans-serif;
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0px;
        margin-top: -50px;
    }
    .sidebar-sub {
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        color: #8b949e;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 30px;
    }
    
    div[data-testid="stMetric"] {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 11px;
        color: #8b949e;
        margin-bottom: 2px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px;
        color: #ffffff;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 10px;
        font-family: 'JetBrains Mono', monospace;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        border-bottom: 1px solid #30363d;
        margin-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 30px;
        padding: 0 8px;
        font-size: 12px;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    
    .stTextInput input {
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }
    div.stButton > button {
        width: 100%;
        background-color: #238636;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 8px;
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .news-card {
        background-color: #1c2128;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
        margin-bottom: 10px;
    }
    .news-title {
        color: #fff;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .news-meta {
        color: #8b949e;
        font-size: 11px;
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">StockLab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">INSTITUTIONAL ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    user_input = st.text_input("ASSET TICKER", "RELIANCE").upper()
    years = st.slider("HISTORY (YRS)", 1, 10, 5)
    prediction_days = st.slider("FORECAST (DAYS)", 1, 7, 1)
    
    st.markdown("---")
    if st.button("INITIALIZE SYSTEM", type="primary"):
        st.session_state['run_analysis'] = True
    else:
        if 'run_analysis' not in st.session_state:
            st.session_state['run_analysis'] = False

def get_currency_symbol(currency_code):
    symbols = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥"}
    return symbols.get(currency_code, currency_code + " ")

def search_global_market(query):
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "quotesCount": 1, "newsCount": 0}
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, params=params, headers=headers)
        data = r.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            best = data['quotes'][0]
            return best['symbol'], best.get('longname', query)
    except:
        pass
    return query, query

@st.cache_data
def get_stock_data(user_query, years):
    start = (date.today() - timedelta(days=years*365)).strftime("%Y-%m-%d")
    end = date.today().strftime("%Y-%m-%d")
    
    ticker_found = None
    company_name = None
    currency_code = "USD"
    
    potential_indian_ticker = f"{user_query.replace(' ', '')}.NS"
    check_indian = yf.Ticker(potential_indian_ticker)
    try:
        hist = check_indian.history(period="1d")
        if not hist.empty:
            ticker_found = potential_indian_ticker
            company_name = check_indian.info.get('longName', user_query)
            currency_code = "INR"
    except:
        pass

    if not ticker_found:
        ticker_found, company_name = search_global_market(user_query)
        if ticker_found.endswith(".NS") or ticker_found.endswith(".BO"):
            currency_code = "INR"
        else:
            try:
                t = yf.Ticker(ticker_found)
                currency_code = t.info.get('currency', 'USD')
            except:
                currency_code = 'USD'

    if ticker_found:
        final_ticker_obj = yf.Ticker(ticker_found)
        raw_data = final_ticker_obj.history(start=start, end=end)
        
        if raw_data.empty:
            return None, None, None, None
            
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
            
        raw_data.reset_index(inplace=True)
        raw_data['Date'] = pd.to_datetime(raw_data['Date']).dt.date
        raw_data.set_index('Date', inplace=True)
        return raw_data, currency_code, ticker_found, company_name
    
    return None, None, None, None

def calculate_technical_indicators(df, prediction_days):
    data = df.copy()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Std_Dev'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['SMA_20'] + (data['Std_Dev'] * 2)
    data['Lower_Band'] = data['SMA_20'] - (data['Std_Dev'] * 2)
    
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    data['Target_Return'] = data['Close'].pct_change().shift(-prediction_days)
    
    data.dropna(inplace=True)
    return data

if st.session_state['run_analysis']:
    with st.spinner(f"CONNECTING TO {user_input}..."):
        df_raw, currency_code, resolved_ticker, company_name = get_stock_data(user_input, years)
        
    if df_raw is None or df_raw.empty:
        st.error(f"[ERROR] Ticker '{user_input}' not found.")
    else:
        currency_symbol = get_currency_symbol(currency_code)
        
        df_quant = calculate_technical_indicators(df_raw, prediction_days)
        features = ['RSI', 'MACD', 'Signal_Line', 'OBV', 'Upper_Band', 'Lower_Band']
        X = df_quant[features]
        y = df_quant['Target_Return']
        
        split = int(len(X) * 0.85)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        model.fit(X_train, y_train)
        
        current_features = X.iloc[[-1]]
        predicted_return = model.predict(current_features)[0]
        current_price = df_quant['Close'].iloc[-1]
        future_price = current_price * (1 + predicted_return)
        price_diff = future_price - current_price
        
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <h3 style="margin:0; padding:0;">{company_name}</h3>
            <span style="color: #8b949e; font-size: 14px;">{resolved_ticker}</span>
        </div>
        """, unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PRICE", f"{currency_symbol}{current_price:,.2f}")
        m2.metric("FORECAST", f"{predicted_return*100:.2f}%", delta_color="normal" if predicted_return > 0 else "inverse")
        m3.metric("TARGET", f"{currency_symbol}{future_price:,.2f}", delta=f"{price_diff:,.2f}")
        
        rsi_val = current_features['RSI'].values[0]
        rsi_sig = "SELL" if rsi_val > 70 else "BUY" if rsi_val < 30 else "HOLD"
        m4.metric(f"RSI ({rsi_val:.0f})", rsi_sig)

        tab1, tab2, tab3, tab4 = st.tabs(["PRICE", "MOMENTUM", "DATA", "NEWS AI"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_quant.index, y=df_quant['Close'], mode='lines', name='PRICE', 
                                     line=dict(color='#00F0FF', width=2)))
            fig.add_trace(go.Scatter(x=df_quant.index, y=df_quant['Upper_Band'], mode='lines', name='UPPER', 
                                     line=dict(color='rgba(0, 255, 0, 0.4)', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df_quant.index, y=df_quant['Lower_Band'], mode='lines', name='LOWER', 
                                     line=dict(color='rgba(255, 0, 0, 0.4)', width=1, dash='dot'), 
                                     fill='tonexty', fillcolor='rgba(0, 255, 0, 0.05)'))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=500,
                xaxis_title=None,
                yaxis_title=None,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(showgrid=True, gridcolor='#222', zeroline=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                                 row_heights=[0.6, 0.4])
            fig2.add_trace(go.Scatter(x=df_quant.index, y=df_quant['MACD'], name='MACD', line=dict(color='#00F0FF')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=df_quant.index, y=df_quant['Signal_Line'], name='SIG', line=dict(color='#FFA500')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=df_quant.index, y=df_quant['RSI'], name='RSI', line=dict(color='#DDA0DD')), row=2, col=1)
            fig2.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig2.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
            
            fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               margin=dict(l=0, r=0, t=10, b=0), height=500, hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)
            
        with tab3:
            st.dataframe(df_quant.tail(50), use_container_width=True, height=500)
            
        with tab4:
            st.markdown("#### AI SENTIMENT ANALYSIS")
            try:
                sn = StockNews([resolved_ticker], save_news=False)
                df_news = sn.read_rss()
                
                if df_news is None or df_news.empty:
                    st.warning("No recent news articles found for this ticker.")
                else:
                    sia = SentimentIntensityAnalyzer()
                    
                    def get_sentiment_score(title):
                        return sia.polarity_scores(title)['compound']
                    
                    df_news['AI_Score'] = df_news['title'].apply(get_sentiment_score)
                    
                    sentiment_score = df_news['AI_Score'].mean()
                    
                    if sentiment_score > 0.15:
                        overall_sentiment = "POSITIVE (BULLISH)"
                        s_color = "#00FF00"
                    elif sentiment_score < -0.15:
                        overall_sentiment = "NEGATIVE (BEARISH)"
                        s_color = "#FF0000"
                    else:
                        overall_sentiment = "NEUTRAL"
                        s_color = "#FFA500"
                    
                    st.markdown(f"""
                    <div style="background-color: #1c2128; padding: 20px; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 20px; text-align: center;">
                        <h2 style="color: {s_color}; margin:0;">{overall_sentiment}</h2>
                        <p style="color: #8b949e; margin:0;">VADER AI Compound Score: {sentiment_score:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i in range(min(5, len(df_news))):
                        item = df_news.iloc[i]
                        news_sentiment = item['AI_Score']
                        
                        if news_sentiment > 0.1: title_color = "#00FF00" 
                        elif news_sentiment < -0.1: title_color = "#FF0000"
                        else: title_color = "#e0e0e0"
                            
                        st.markdown(f"""
                        <div class="news-card">
                            <div class="news-title" style="color: {title_color};">{item['title']}</div>
                            <div class="news-meta">{item['published']} | Sentiment: {news_sentiment:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"AI SYSTEM ERROR: {str(e)}")

else:
    st.markdown("""
    <div style='display: flex; justify-content: center; align-items: center; height: 60vh; color: #444; font-family: sans-serif;'>
        <h3>SELECT ASSET & INITIALIZE</h3>
    </div>
    """, unsafe_allow_html=True)