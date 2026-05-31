import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import requests
from threading import Lock

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
lock = Lock()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "stock_history.csv")

STOCK_LIST = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "AXISBANK", "LT", "WIPRO", "MARUTI", "SUNPHARMA"]

# ------------------------------
# DATA LOADING (from CSV or live)
# ------------------------------
def load_stock_data():
    """Load historical data from CSV (created locally)."""
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df
    else:
        logging.warning("No stock_history.csv found. Please seed it once.")
        return pd.DataFrame(columns=['Date','Symbol','Close','Open','High','Low','Volume'])

def update_history():
    """Placeholder: in production, you would refresh CSV via external cron."""
    # For Render, we rely on pre‑seeded CSV. No live fetch.
    pass

# ------------------------------
# TECHNICAL INDICATORS (same as before)
# ------------------------------
def compute_indicators(df):
    df = df.sort_values('Date').copy()
    for w in [5,10,20,50,200]:
        df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std()
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']
    return df

def predict_stock(symbol, df):
    if df.empty or len(df) < 50:
        return None, None
    df = compute_indicators(df).dropna()
    if len(df) < 30:
        return None, None
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    features = ['RSI', 'MACD', 'volatility', 'volume_ratio', 'SMA_20', 'SMA_50']
    X = df[features]
    y = df['Close'].shift(-60)
    valid = y.notna()
    X, y = X[valid], y[valid]
    if len(X) < 30:
        return None, None
    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    last_price = df['Close'].iloc[-1]
    current_features = X.iloc[-1:][features]
    pred_future = model.predict(current_features)[0]
    profit_pct = (pred_future - last_price) / last_price * 100
    confidence = max(0, min(100, 100 - (mae / last_price) * 100))
    return profit_pct, confidence

def get_opportunity_stocks():
    df_all = load_stock_data()
    if df_all.empty:
        return []
    opportunities = []
    for sym in STOCK_LIST:
        sym_df = df_all[df_all['Symbol'] == sym].copy()
        if sym_df.empty:
            continue
        df_indic = compute_indicators(sym_df).dropna()
        if df_indic.empty:
            continue
        latest = df_indic.iloc[-1]
        score = 0
        if latest['RSI'] < 30:
            score += 2
        if latest['EMA_20'] > latest['EMA_50']:
            score += 1
        if latest['volume_ratio'] > 1.2:
            score += 1
        if score >= 2:
            profit, conf = predict_stock(sym, sym_df)
            if profit and profit > 0:
                opportunities.append({
                    'symbol': sym,
                    'current_price': round(latest['Close'], 2),
                    'predicted_profit_pct': round(profit, 1),
                    'confidence': round(conf, 1),
                    'local_low': latest['RSI'] < 30
                })
    opportunities.sort(key=lambda x: x['confidence'], reverse=True)
    return opportunities[:10]

# ------------------------------
# MUTUAL FUNDS (same as before, working)
# ------------------------------
def get_mutual_fund_history(fund_code):
    url = f"https://api.mfapi.in/mf/{fund_code}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if 'data' not in data:
            return None
        history = []
        for entry in data['data']:
            try:
                nav = float(entry['nav'])
                dt = datetime.strptime(entry['date'], '%d-%m-%Y')
                history.append({'date': dt, 'nav': nav})
            except:
                continue
        history.sort(key=lambda x: x['date'])
        return {'name': data['meta']['scheme_name'], 'code': fund_code, 'history': history}
    except Exception as e:
        logging.error(f"MF error {fund_code}: {e}")
        return None

def compute_return(history, days_back):
    if not history or len(history) < days_back + 1:
        return None
    latest = history[-1]['nav']
    past = history[-1 - days_back]['nav']
    return (latest - past) / past * 100

def get_mutual_funds_with_ranges(period):
    period_map = {'daily':1, 'weekly':7, 'monthly':30, '3year':756, '5year':1260}
    days = period_map.get(period, 1)
    fund_codes = ["118531", "119551", "120638", "122639", "100770", "118483"]
    results = []
    for code in fund_codes:
        fund_data = get_mutual_fund_history(code)
        if not fund_data:
            continue
        ret = compute_return(fund_data['history'], days)
        if ret is not None:
            results.append({
                'code': code,
                'name': fund_data['name'],
                'nav': fund_data['history'][-1]['nav'],
                'return_pct': round(ret, 2)
            })
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    return results[:5], results[-5:][::-1]

def investment_snapshot(symbol, amount):
    df_all = load_stock_data()
    sym_df = df_all[df_all['Symbol'] == symbol].copy()
    if sym_df.empty:
        return {"error": "No data for this stock"}
    profit_pct, confidence = predict_stock(symbol, sym_df)
    if profit_pct is None:
        return {"error": "Insufficient data for prediction"}
    profit_amount = amount * profit_pct / 100
    return {
        "symbol": symbol,
        "amount": amount,
        "predicted_profit_pct": round(profit_pct, 1),
        "predicted_profit_amount": round(profit_amount, 2),
        "confidence": round(confidence, 1),
        "advice": f"Invest ₹{amount} in {symbol}. Expected profit ~ ₹{round(profit_amount,2)} with {round(confidence,1)}% confidence."
    }

# ------------------------------
# FLASK ROUTES (unchanged)
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/opportunities')
def api_opportunities():
    with lock:
        stocks = get_opportunity_stocks()
    return jsonify(stocks)

@app.route('/api/mutual_funds')
def api_mutual_funds():
    period = request.args.get('period', 'daily')
    gainers, losers = get_mutual_funds_with_ranges(period)
    return jsonify({"gainers": gainers, "losers": losers, "period": period})

@app.route('/api/invest', methods=['POST'])
def api_invest():
    data = request.get_json()
    symbol = data.get('symbol')
    amount = float(data.get('amount', 0))
    if not symbol or amount <= 0:
        return jsonify({"error": "Invalid input"}), 400
    result = investment_snapshot(symbol, amount)
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
