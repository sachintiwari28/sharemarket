import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import requests
from threading import Lock

# ------------------------------
# 1. Setup & Configuration
# ------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
lock = Lock()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "stock_history.csv")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.json")

# Nifty 200 stocks (subset for demo – you can expand)
STOCK_LIST = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "AXISBANK", "LT", "WIPRO", "MARUTI", "SUNPHARMA"
]

# ------------------------------
# 2. Data Scraper (nsepython – free, works for India)
# ------------------------------
def fetch_stock_data():
    """Fetch latest stock data using nsepython (no API key)."""
    try:
        from nsepython import nse_eq
    except ImportError:
        logging.error("nsepython not installed. Install with: pip install nsepython")
        return pd.DataFrame()

    all_data = []
    for symbol in STOCK_LIST:
        try:
            data = nse_eq(symbol)
            if data and 'lastPrice' in data:
                row = {
                    'Symbol': symbol,
                    'Date': datetime.now().date(),
                    'Open': data.get('open', 0),
                    'High': data.get('dayHigh', 0),
                    'Low': data.get('dayLow', 0),
                    'Close': data.get('lastPrice', 0),
                    'Volume': data.get('totalTradedVolume', 0)
                }
                all_data.append(row)
                logging.info(f"Fetched {symbol}: ₹{row['Close']}")
            else:
                logging.warning(f"No data for {symbol}")
        except Exception as e:
            logging.error(f"Failed {symbol}: {e}")
    return pd.DataFrame(all_data)

def update_history():
    """Append new data to CSV (keep last 1 year)."""
    new_data = fetch_stock_data()
    if new_data.empty:
        logging.warning("No new data fetched.")
        if not os.path.exists(HISTORY_FILE):
            pd.DataFrame(columns=['Date','Symbol','Close','Open','High','Low','Volume']).to_csv(HISTORY_FILE, index=False)
        return
    try:
        old = pd.read_csv(HISTORY_FILE)
        old['Date'] = pd.to_datetime(old['Date']).dt.date
        combined = pd.concat([old, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date','Symbol'], keep='last')
        cutoff = datetime.now().date() - timedelta(days=365)
        combined = combined[pd.to_datetime(combined['Date']).dt.date >= cutoff]
        combined.to_csv(HISTORY_FILE, index=False)
        logging.info(f"History updated, {len(combined)} records.")
    except FileNotFoundError:
        new_data.to_csv(HISTORY_FILE, index=False)
        logging.info("Created new history file.")

# ------------------------------
# 3. Technical Indicators (for screener & model)
# ------------------------------
def compute_indicators(df):
    """Add technical indicators: SMA, EMA, RSI, MACD, volatility, etc."""
    df = df.sort_values('Date').copy()
    # Simple moving averages
    for w in [5,10,20,50,200]:
        df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Volatility
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std()
    # Volume ratio
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']
    return df

# ------------------------------
# 4. Prediction Model (fallback to linear regression if DL libs missing)
# ------------------------------
def predict_with_lstm(df, symbol):
    """Use LSTM + LightGBM if available, else linear regression."""
    try:
        # Try to import deep learning libraries
        import torch
        import torch.nn as nn
        import lightgbm as lgb
        from sklearn.preprocessing import MinMaxScaler
        USE_DL = True
    except ImportError:
        USE_DL = False
        logging.warning("PyTorch or LightGBM not installed. Using linear regression fallback.")

    if not USE_DL or len(df) < 100:
        # Fallback to simple linear regression (as in original app)
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error
        df = compute_indicators(df).dropna()
        if len(df) < 50:
            return None, None
        features = ['RSI', 'MACD', 'volatility', 'volume_ratio', f'SMA_20', f'SMA_50']
        X = df[features]
        y = df['Close'].shift(-60)  # predict 60 days ahead
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

    # ------------------------------
    # Deep learning branch (LSTM + LightGBM)
    # ------------------------------
    # Prepare features
    df = compute_indicators(df).dropna()
    feature_cols = ['RSI', 'MACD', 'volatility', 'volume_ratio', 'SMA_20', 'SMA_50', 'returns']
    target = 'Close'
    X = df[feature_cols].values
    y = df[target].values.reshape(-1,1)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    seq_len = 60
    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y_scaled[i+seq_len])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    # Simple LSTM model
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_lstm = SimpleLSTM(X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
    # Train
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    for epoch in range(30):
        model_lstm.train()
        optimizer.zero_grad()
        outputs = model_lstm(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    # LightGBM
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1)
    lgb_model.fit(X_train_flat, y_train.ravel())
    # Predict
    last_seq = X_scaled[-seq_len:].reshape(1, seq_len, -1)
    with torch.no_grad():
        lstm_pred_scaled = model_lstm(torch.FloatTensor(last_seq).to(device)).cpu().numpy()
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)[0,0]
    lgb_pred_scaled = lgb_model.predict(last_seq.reshape(1, -1))[0]
    lgb_pred = scaler_y.inverse_transform([[lgb_pred_scaled]])[0,0]
    ensemble = 0.6 * lstm_pred + 0.4 * lgb_pred
    last_price = df['Close'].iloc[-1]
    profit_pct = (ensemble - last_price) / last_price * 100
    # Estimate confidence from validation error
    y_test_actual = scaler_y.inverse_transform(y_test)
    y_test_pred = scaler_y.inverse_transform(lgb_model.predict(X_test_flat).reshape(-1,1))
    mape = np.mean(np.abs((y_test_actual - y_test_pred) / y_test_actual)) * 100
    confidence = max(0, min(100, 100 - mape))
    return profit_pct, confidence

# ------------------------------
# 5. Stock Screener (Multi-Criteria)
# ------------------------------
def screen_stock(symbol, df):
    if df.empty or len(df) < 50:
        return None
    df = compute_indicators(df).dropna()
    if df.empty:
        return None
    latest = df.iloc[-1]
    # Scoring
    score = 0
    # momentum (RSI)
    if latest['RSI'] < 30:
        score += 2
    elif latest['RSI'] > 70:
        score -= 1
    # trend (golden cross)
    if latest['EMA_20'] > latest['EMA_50']:
        score += 1
    else:
        score -= 1
    # volume surge
    if latest['volume_ratio'] > 1.2:
        score += 1
    # volatility (low is good)
    if latest['volatility'] < 0.02 * latest['Close']:
        score += 1
    return {
        'symbol': symbol,
        'current_price': latest['Close'],
        'score': score,
        'rsi': latest['RSI'],
        'volatility': latest['volatility'],
        'trend_up': latest['EMA_20'] > latest['EMA_50']
    }

def get_opportunity_stocks():
    """Run screener on all stocks and return top opportunities."""
    update_history()  # ensure latest data
    try:
        df_all = pd.read_csv(HISTORY_FILE)
    except FileNotFoundError:
        return []
    opportunities = []
    for sym in STOCK_LIST:
        sym_df = df_all[df_all['Symbol'] == sym].copy()
        if sym_df.empty:
            continue
        result = screen_stock(sym, sym_df)
        if result and result['score'] >= 1:  # only promising ones
            profit, conf = predict_with_lstm(sym_df, sym)
            if profit is not None and profit > 0:
                opportunities.append({
                    'symbol': sym,
                    'current_price': round(result['current_price'], 2),
                    'predicted_profit_pct': round(profit, 1),
                    'confidence': round(conf, 1),
                    'local_low': result['rsi'] < 30
                })
    opportunities.sort(key=lambda x: x['confidence'] * x['predicted_profit_pct'], reverse=True)
    return opportunities[:10]

# ------------------------------
# 6. Mutual Funds (mfapi.in)
# ------------------------------
def get_mutual_funds():
    """Fetch top gainers/losers from mfapi.in (Indian mutual funds)."""
    fund_codes = [
        "118531", "119551", "120638", "122639", "100770", "118483"
    ]
    results = []
    for code in fund_codes:
        try:
            url = f"https://api.mfapi.in/mf/{code}"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if 'data' not in data or len(data['data']) < 7:
                continue
            latest = data['data'][0]
            nav_now = float(latest['nav'])
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
    results.sort(key=lambda x: x['week_return'], reverse=True)
    return results[:5], results[-5:][::-1]

# ------------------------------
# 7. Investment Snapshot
# ------------------------------
def investment_snapshot(symbol, amount):
    try:
        df_all = pd.read_csv(HISTORY_FILE)
        sym_df = df_all[df_all['Symbol'] == symbol].copy()
        if sym_df.empty:
            return {"error": "No data for this stock"}
        profit_pct, confidence = predict_with_lstm(sym_df, symbol)
        if profit_pct is None:
            return {"error": "Prediction failed – insufficient data"}
        profit_amount = amount * profit_pct / 100
        return {
            "symbol": symbol,
            "amount": amount,
            "predicted_profit_pct": round(profit_pct, 1),
            "predicted_profit_amount": round(profit_amount, 2),
            "confidence": round(confidence, 1),
            "advice": f"Invest ₹{amount} in {symbol}. Expected profit ~ ₹{round(profit_amount,2)} with {round(confidence,1)}% confidence."
        }
    except Exception as e:
        return {"error": str(e)}

# ------------------------------
# 8. Flask Routes
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

# ------------------------------
# 9. Scheduled Updates (for Render Cron)
# ------------------------------
def scheduled_update():
    with lock:
        update_history()
        logging.info("Scheduled data update completed.")

# ------------------------------
# 10. Run the App
# ------------------------------
if __name__ == '__main__':
    # Initial data fetch
    update_history()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
