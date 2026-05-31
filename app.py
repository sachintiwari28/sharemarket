import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ================= CONFIG =================
STOCK_LIST = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
              "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
              "AXISBANK.NS", "LT.NS", "WIPRO.NS", "MARUTI.NS", "SUNPHARMA.NS"]
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "stock_history.csv")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.json")

# ========== HELPER: fetch & update stock data ==========
def fetch_stock_data():
    """Download latest 1y data for all stocks, return DataFrame"""
    all_data = []
    for symbol in STOCK_LIST:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            hist['Symbol'] = symbol
            all_data.append(hist)
        except Exception as e:
            logging.error(f"Failed {symbol}: {e}")
    if not all_data:
        return pd.DataFrame()
    df = pd.concat(all_data)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

def update_history():
    """Append new data to CSV (keep last 1y)"""
    new_data = fetch_stock_data()
    if new_data.empty:
        return
    try:
        old = pd.read_csv(HISTORY_FILE)
        old['Date'] = pd.to_datetime(old['Date']).dt.date
        combined = pd.concat([old, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
        # keep only last 365 days
        cutoff = datetime.now().date() - timedelta(days=365)
        combined = combined[pd.to_datetime(combined['Date']).dt.date >= cutoff]
        combined.to_csv(HISTORY_FILE, index=False)
    except FileNotFoundError:
        new_data.to_csv(HISTORY_FILE, index=False)

# ========== STOCK SCREENER (localized MA + growth) ==========
def compute_technical_features(df):
    """Add moving averages, RSI, and undervalued signal"""
    df = df.sort_values('Date')
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    # RSI (14 days)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # Localized low: price below 20-day SMA but above 200-day SMA
    df['local_low'] = (df['Close'] < df['SMA_20']) & (df['Close'] > df['SMA_200'])
    return df

def predict_growth(symbol, days_ahead=60):
    """Simple linear regression using price to SMA ratio, RSI, day of week"""
    df = pd.read_csv(HISTORY_FILE)
    df = df[df['Symbol'] == symbol].copy()
    if len(df) < 100:
        return None, None
    df = compute_technical_features(df)
    df = df.dropna()
    # features
    df['price_sma20_ratio'] = df['Close'] / df['SMA_20']
    df['price_sma50_ratio'] = df['Close'] / df['SMA_50']
    df['dayofweek'] = pd.to_datetime(df['Date']).dt.dayofweek
    features = ['price_sma20_ratio', 'price_sma50_ratio', 'RSI', 'dayofweek']
    X = df[features]
    y = df['Close'].shift(-days_ahead)  # future price
    # drop last rows where future unknown
    valid = y.notna()
    X = X[valid]
    y = y[valid]
    if len(X) < 50:
        return None, None
    # train/test split (time series order)
    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    # confidence based on MAE on test set
    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    last_price = df['Close'].iloc[-1]
    current_features = X.iloc[-1:][features]
    pred_future = model.predict(current_features)[0]
    profit_pct = (pred_future - last_price) / last_price * 100
    # confidence: 100 - (mae / last_price)*100, capped
    confidence = max(0, min(100, 100 - (mae / last_price) * 100))
    return profit_pct, confidence

def get_opportunity_stocks():
    """Return list of stocks that are undervalued (local low) + positive predicted growth"""
    update_history()  # ensure latest data
    df = pd.read_csv(HISTORY_FILE)
    opportunities = []
    for symbol in STOCK_LIST:
        symbol_df = df[df['Symbol'] == symbol].sort_values('Date')
        if symbol_df.empty:
            continue
        symbol_df = compute_technical_features(symbol_df)
        latest = symbol_df.iloc[-1]
        if latest['local_low']:
            profit, conf = predict_growth(symbol)
            if profit and profit > 0:
                opportunities.append({
                    'symbol': symbol,
                    'current_price': round(latest['Close'], 2),
                    'predicted_profit_pct': round(profit, 1),
                    'confidence': round(conf, 1),
                    'local_low': True
                })
    # sort by confidence * profit
    opportunities.sort(key=lambda x: x['confidence'] * x['predicted_profit_pct'], reverse=True)
    return opportunities[:10]

# ========== MUTUAL FUNDS (top gainers/losers) ==========
def get_mutual_funds():
    """Fetch from mfapi.in, calculate 1-week return and top/bottom"""
    # Use a fixed set of popular funds (can be expanded)
    fund_codes = [
        "119551", "119552", "119553",  # SBI Bluechip, etc. (example)
        "120638", "120639", "120640",
        "118531", "118532"
    ]
    results = []
    for code in fund_codes:
        try:
            url = f"https://api.mfapi.in/mf/{code}"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if 'data' not in data or len(data['data']) < 7:
                continue
            # latest NAV
            latest = data['data'][0]
            nav_now = float(latest['nav'])
            # 7 days ago NAV
            week_ago = data['data'][7]
            nav_week_ago = float(week_ago['nav'])
            week_return = (nav_now - nav_week_ago) / nav_week_ago * 100
            results.append({
                'code': code,
                'name': data['meta']['scheme_name'],
                'nav': nav_now,
                'week_return': round(week_return, 2)
            })
        except Exception as e:
            logging.error(f"MF {code} error: {e}")
    # top 5 gainers and losers
    results.sort(key=lambda x: x['week_return'], reverse=True)
    top_gainers = results[:5]
    top_losers = results[-5:][::-1]  # worst first
    return top_gainers, top_losers

# ========== INVESTMENT SNAPSHOT ==========
def investment_snapshot(symbol, amount):
    profit_pct, confidence = predict_growth(symbol)
    if profit_pct is None:
        return {"error": "Not enough data for prediction"}
    profit_amount = amount * profit_pct / 100
    return {
        "symbol": symbol,
        "amount": amount,
        "predicted_profit_pct": profit_pct,
        "predicted_profit_amount": round(profit_amount, 2),
        "confidence": confidence,
        "advice": f"Invest ₹{amount} in {symbol}. Expected profit ~ ₹{round(profit_amount,2)} with {confidence}% confidence."
    }

# ========== FLASK ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/opportunities')
def api_opportunities():
    stocks = get_opportunity_stocks()
    return jsonify(stocks)

@app.route('/api/mutual_funds')
def api_mutual_funds():
    gainers, losers = get_mutual_funds()
    return jsonify({"gainers": gainers, "losers": losers})

@app.route('/api/invest', methods=['POST'])
def api_invest():
    data = request.get_json()
    symbol = data.get('symbol')
    amount = float(data.get('amount', 0))
    if not symbol or amount <= 0:
        return jsonify({"error": "Invalid input"}), 400
    result = investment_snapshot(symbol, amount)
    return jsonify(result)

# Run scheduler for daily update (APScheduler)
from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_history, trigger="cron", hour=18, minute=0)  # 6 PM daily
scheduler.start()

if __name__ == '__main__':
    # initial data load
    update_history()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
