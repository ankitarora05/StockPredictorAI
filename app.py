# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
import os
import threading

warnings.filterwarnings("ignore")

# =========================================================
# 0. GLOBAL CONFIG
# =========================================================
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "9984.T", "6758.T", "7203.T", "8306.T",
    "0700.HK", "9988.HK", "0005.HK",
    "600036.SS", "600519.SS",
    "005930.KS", "000660.KS",
    "D05.SI", "O39.SI"
]

MODEL_FILE = "stock_model.joblib"
SCALER_FILE = "scaler.joblib"
FEATURES_FILE = "features.joblib"
TARGETS_FILE = "target_cols.joblib"

# Global state
model = None
scaler = None
feature_names = None
target_cols = None
training_in_progress = False
training_status = "idle"
training_message = ""

# =========================================================
# 1. TRAINING FUNCTIONS (unchanged)
# =========================================================
def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            col_map['open'] = col
        elif 'high' in col_lower:
            col_map['high'] = col
        elif 'low' in col_lower:
            col_map['low'] = col
        elif 'close' in col_lower:
            col_map['close'] = col
        elif 'volume' in col_lower:
            col_map['volume'] = col
    if len(col_map) < 5:
        return None
    df = df[list(col_map.values())]
    df.columns = list(col_map.keys())
    return df

def compute_features(data):
    df_feat = data.copy()
    for lag in [1, 5, 10, 21]:
        df_feat[f'close_lag_{lag}'] = df_feat['close'].shift(lag)
        df_feat[f'volume_lag_{lag}'] = df_feat['volume'].shift(lag)
    for window in [5, 10, 21]:
        df_feat[f'close_ma_{window}'] = df_feat['close'].rolling(window).mean()
        df_feat[f'close_std_{window}'] = df_feat['close'].rolling(window).std()
        df_feat[f'volume_ma_{window}'] = df_feat['volume'].rolling(window).mean()
    df_feat['high_low_ratio'] = df_feat['high'] / df_feat['low']
    df_feat['close_open_ratio'] = df_feat['close'] / df_feat['open']
    df_feat['daily_return'] = df_feat['close'].pct_change()
    df_feat['return_lag_1'] = df_feat['daily_return'].shift(1)
    for i in range(1, 6):
        df_feat[f'target_{i}d'] = (df_feat['close'].shift(-i) / df_feat['close']) - 1
    return df_feat

def train_model_thread():
    global training_in_progress, training_status, training_message
    global model, scaler, feature_names, target_cols

    try:
        training_in_progress = True
        training_status = "training"
        training_message = "Downloading stock data (this may take a few minutes)..."

        all_data = []
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = "2015-01-01"

        for idx, ticker in enumerate(TICKERS):
            training_message = f"Fetching {ticker} ({idx+1}/{len(TICKERS)})..."
            print(f"  Fetching {ticker}...")
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                df = clean_columns(df)
                if df is None or len(df) < 100:
                    continue
                df_feat = compute_features(df)
                df_feat = df_feat.dropna()
                if len(df_feat) > 0:
                    all_data.append(df_feat)
            except Exception as e:
                print(f"    Skipping {ticker}: {e}")

        if not all_data:
            raise RuntimeError("No data collected. Check your internet connection and ticker list.")

        training_message = "Combining and preprocessing data..."
        combined = pd.concat(all_data, axis=0, ignore_index=True)
        print(f"✅ Total samples collected: {len(combined)}")

        target_cols_local = ['target_1d', 'target_2d', 'target_3d', 'target_4d', 'target_5d']
        features_local = [col for col in combined.columns if col not in target_cols_local]
        X = combined[features_local]
        y = combined[target_cols_local]

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        training_message = "Scaling features and training Random Forest..."
        scaler_local = StandardScaler()
        X_train_scaled = scaler_local.fit_transform(X_train)
        X_test_scaled = scaler_local.transform(X_test)

        model_local = RandomForestRegressor(n_estimators=300,max_depth=25,min_samples_split=10,random_state=42,n_jobs=-1)
        model_local.fit(X_train_scaled, y_train)

        y_pred = model_local.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        avg_mae = np.mean(mae)
        print(f"✅ Avg MAE (returns): {avg_mae:.4f}")

        joblib.dump(model_local, MODEL_FILE)
        joblib.dump(scaler_local, SCALER_FILE)
        joblib.dump(features_local, FEATURES_FILE)
        joblib.dump(target_cols_local, TARGETS_FILE)

        model = model_local
        scaler = scaler_local
        feature_names = features_local
        target_cols = target_cols_local

        training_status = "done"
        training_message = f"✅ Training complete! Average error: {avg_mae:.2%}"

    except Exception as e:
        training_status = "error"
        training_message = f"❌ Training failed: {str(e)}"
        print(f"❌ Training error: {e}")

    finally:
        training_in_progress = False

def load_artifacts():
    if all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, FEATURES_FILE, TARGETS_FILE]):
        model_local = joblib.load(MODEL_FILE)
        scaler_local = joblib.load(SCALER_FILE)
        features_local = joblib.load(FEATURES_FILE)
        target_cols_local = joblib.load(TARGETS_FILE)
        print("✅ Artifacts loaded from disk.")
        return model_local, scaler_local, features_local, target_cols_local
    else:
        print("⚠️ No existing artifacts found. Use 'Train Model' button to train.")
        return None, None, None, None

# =========================================================
# 2. FASTAPI APP
# =========================================================
app = FastAPI(
    title="Stock Predictor AI – Global Tracker",
    description="Train on demand, then predict the next 5 closing prices for any global stock.",
    version="3.0"
)

model, scaler, feature_names, target_cols = load_artifacts()

# =========================================================
# 3. HELPER FUNCTIONS FOR API (FIXED)
# =========================================================
SUFFIX_MAP = {
    "US": "",
    "NSE": ".NS",
    "BSE": ".BO",
    "T": ".T",
    "HK": ".HK",
    "SI": ".SI",
    "SS": ".SS",
    "SZ": ".SZ",
    "KS": ".KS"
}

EXCHANGE_NAMES = {
    "US": "US (NYSE/NASDAQ)",
    "NSE": "India (NSE)",
    "BSE": "India (BSE)",
    "T": "Japan (Tokyo)",
    "HK": "Hong Kong",
    "SI": "Singapore",
    "SS": "China (Shanghai)",
    "SZ": "China (Shenzhen)",
    "KS": "South Korea (KOSPI)"
}

def fetch_stock_data(symbol, exchange="US"):
    """
    Fetches data for a stock, ensuring we get a valid recent close price.
    """
    suffix = SUFFIX_MAP.get(exchange, "")
    if suffix:
        # Remove any existing suffix to avoid duplication
        for s in SUFFIX_MAP.values():
            if s and symbol.endswith(s):
                symbol = symbol[:-len(s)]
        symbol = symbol + suffix

    end_date = datetime.today().strftime('%Y-%m-%d')
    used_symbol = symbol
    df = None
    found = False

    # Try multiple lookback periods
    for days_back in [90, 120, 180]:
        if found:
            break
        start_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        try:
            temp_df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            temp_df = clean_columns(temp_df)
            if temp_df is None or len(temp_df) < 30:
                continue
            # Check if the last close is finite
            if not np.isfinite(temp_df['close'].iloc[-1]):
                # Try to find the last finite close
                last_finite = temp_df['close'].dropna()
                if len(last_finite) == 0:
                    continue
                # If the last row is NaN but others are fine, we might truncate
                # We'll just drop rows with NaN close and take the last valid
                temp_df = temp_df[temp_df['close'].notna()]
                if len(temp_df) < 30:
                    continue
            df = temp_df
            used_symbol = symbol
            found = True
            break
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            continue

    # Fallback: try without suffix (if user provided it)
    if not found and suffix:
        base_symbol = symbol.replace(suffix, "")
        for days_back in [90, 120, 180]:
            if found:
                break
            start_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            try:
                temp_df = yf.download(base_symbol, start=start_date, end=end_date, progress=False)
                temp_df = clean_columns(temp_df)
                if temp_df is None or len(temp_df) < 30:
                    continue
                if not np.isfinite(temp_df['close'].iloc[-1]):
                    last_finite = temp_df['close'].dropna()
                    if len(last_finite) == 0:
                        continue
                    temp_df = temp_df[temp_df['close'].notna()]
                    if len(temp_df) < 30:
                        continue
                df = temp_df
                used_symbol = base_symbol
                found = True
                break
            except Exception:
                continue

    if not found or df is None or len(df) < 30:
        exchange_name = EXCHANGE_NAMES.get(exchange, exchange)
        raise HTTPException(
            status_code=404,
            detail=f"Could not find enough valid price data for '{symbol}' on {exchange_name}. "
                   f"Please check the symbol and exchange."
        )

    # Final check: ensure we have a finite current price
    if not np.isfinite(df['close'].iloc[-1]):
        # Attempt to find last finite value
        last_valid = df['close'].dropna()
        if len(last_valid) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No valid close price found for '{symbol}'. The stock may be delisted or the symbol is incorrect."
            )
        # Use the last valid price and truncate rows beyond that
        df = df.loc[:last_valid.index[-1]]
        if len(df) < 30:
            raise HTTPException(
                status_code=404,
                detail=f"Not enough valid data after filtering. Need at least 30 trading days."
            )

    return df, used_symbol

# =========================================================
# 4. API ENDPOINTS (UI unchanged – identical to previous)
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def ui():
    # ... (copy the full HTML from the previous version)
    # I'll include a shortened placeholder here – you should keep the full HTML from before.
    # For brevity, I'll embed the same UI as before.
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Predictor AI – Global Tracker</title>
        <meta name="description" content="AI-powered global stock price prediction for the next 5 trading days. Select any exchange and get instant forecasts.">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            /* ... same styles as before ... */
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(145deg, #0a0e1a 0%, #1a1f2f 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
                color: #e4e7f0;
            }
            .container {
                max-width: 1000px;
                width: 100%;
                background: rgba(255, 255, 255, 0.04);
                backdrop-filter: blur(20px);
                border-radius: 48px;
                padding: 48px 40px;
                box-shadow: 0 40px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.05) inset;
                border: 1px solid rgba(255,255,255,0.06);
            }
            .header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 16px;
                margin-bottom: 32px;
            }
            .logo {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .logo-icon {
                background: linear-gradient(135deg, #6a5af9, #a855f7);
                width: 48px;
                height: 48px;
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                font-weight: 700;
                color: #fff;
                box-shadow: 0 8px 24px rgba(106, 90, 249, 0.3);
            }
            .logo-text h1 {
                font-size: 24px;
                font-weight: 700;
                letter-spacing: -0.5px;
                background: linear-gradient(to right, #e4e7f0, #a78bfa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .logo-text span {
                font-size: 13px;
                color: #8b8fa6;
                background: rgba(255,255,255,0.06);
                padding: 4px 12px;
                border-radius: 40px;
                margin-left: 8px;
                -webkit-text-fill-color: #8b8fa6;
            }
            .badge-global {
                background: rgba(52, 211, 153, 0.15);
                border: 1px solid rgba(52, 211, 153, 0.25);
                padding: 6px 18px;
                border-radius: 40px;
                font-size: 13px;
                font-weight: 600;
                color: #34d399;
                letter-spacing: 0.3px;
                display: inline-flex;
                align-items: center;
                gap: 6px;
            }
            .badge-global::before { content: "🌍"; font-size: 16px; }
            .status-bar {
                margin: 12px 0 20px 0;
                padding: 12px 20px;
                border-radius: 16px;
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.06);
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 12px;
            }
            .status-bar .msg {
                font-size: 14px;
                color: #8b8fa6;
            }
            .status-bar .msg strong { color: #e4e7f0; }
            .train-btn {
                padding: 10px 24px;
                border-radius: 40px;
                border: none;
                background: linear-gradient(135deg, #f59e0b, #f97316);
                color: #fff;
                font-weight: 600;
                font-size: 14px;
                cursor: pointer;
                transition: all 0.25s;
                font-family: 'Inter', sans-serif;
                box-shadow: 0 4px 16px rgba(245, 158, 11, 0.25);
            }
            .train-btn:hover { transform: scale(1.02); box-shadow: 0 6px 24px rgba(245, 158, 11, 0.4); }
            .train-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
            .train-btn.success { background: linear-gradient(135deg, #10b981, #059669); }
            .train-btn.error { background: linear-gradient(135deg, #ef4444, #dc2626); }
            .search-section {
                margin: 16px 0 32px 0;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .search-box {
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                align-items: center;
            }
            .search-box select {
                padding: 16px 20px;
                border-radius: 60px;
                border: 1px solid rgba(255,255,255,0.08);
                background: rgba(255,255,255,0.06);
                color: #e4e7f0;
                font-size: 14px;
                font-weight: 500;
                outline: none;
                transition: all 0.25s;
                font-family: 'Inter', sans-serif;
                cursor: pointer;
                min-width: 160px;
                appearance: none;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%238b8fa6' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
                background-repeat: no-repeat;
                background-position: right 16px center;
                padding-right: 44px;
            }
            .search-box select:focus {
                border-color: #a855f7;
                box-shadow: 0 0 0 4px rgba(168, 85, 247, 0.15);
            }
            .search-box select option {
                background: #1a1f2f;
                color: #e4e7f0;
            }
            .search-box input {
                flex: 1;
                min-width: 180px;
                padding: 16px 24px;
                border-radius: 60px;
                border: 1px solid rgba(255,255,255,0.08);
                background: rgba(255,255,255,0.06);
                color: #fff;
                font-size: 16px;
                outline: none;
                transition: all 0.25s;
                font-family: 'Inter', sans-serif;
            }
            .search-box input::placeholder { color: #6b7085; }
            .search-box input:focus {
                border-color: #a855f7;
                background: rgba(255,255,255,0.08);
                box-shadow: 0 0 0 4px rgba(168, 85, 247, 0.15);
            }
            .search-box button {
                padding: 16px 36px;
                border-radius: 60px;
                border: none;
                background: linear-gradient(135deg, #6a5af9, #a855f7);
                color: #fff;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.25s;
                font-family: 'Inter', sans-serif;
                box-shadow: 0 8px 24px rgba(106, 90, 249, 0.25);
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .search-box button:hover:not(:disabled) {
                transform: scale(1.02);
                box-shadow: 0 12px 32px rgba(106, 90, 249, 0.4);
            }
            .search-box button:disabled {
                opacity: 0.4;
                cursor: not-allowed;
                transform: none;
            }
            .error-message {
                color: #f87171;
                font-size: 14px;
                padding-left: 8px;
                min-height: 24px;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .result-section {
                margin-top: 24px;
                display: none;
                animation: fadeIn 0.5s ease;
            }
            .result-section.visible { display: block; }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(12px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .stock-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 12px;
                margin-bottom: 24px;
            }
            .stock-name {
                font-size: 28px;
                font-weight: 700;
                letter-spacing: -0.3px;
            }
            .stock-name small {
                font-size: 14px;
                font-weight: 400;
                color: #8b8fa6;
                margin-left: 10px;
            }
            .current-price {
                font-size: 20px;
                background: rgba(255,255,255,0.06);
                padding: 8px 20px;
                border-radius: 40px;
                border: 1px solid rgba(255,255,255,0.06);
            }
            .current-price strong {
                font-weight: 700;
                color: #a78bfa;
            }
            .chart-container {
                background: rgba(0,0,0,0.2);
                border-radius: 24px;
                padding: 24px 20px 12px 20px;
                margin: 20px 0 24px 0;
                border: 1px solid rgba(255,255,255,0.04);
            }
            .prediction-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 12px;
                margin-top: 16px;
            }
            .prediction-card {
                background: rgba(255,255,255,0.04);
                border-radius: 16px;
                padding: 16px 12px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.04);
                transition: all 0.2s;
            }
            .prediction-card:hover {
                background: rgba(255,255,255,0.08);
                transform: translateY(-2px);
            }
            .prediction-card .day {
                font-size: 13px;
                color: #8b8fa6;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .prediction-card .price {
                font-size: 20px;
                font-weight: 700;
                margin-top: 6px;
                color: #e4e7f0;
            }
            .prediction-card .change {
                font-size: 13px;
                font-weight: 600;
                margin-top: 4px;
            }
            .change.positive { color: #34d399; }
            .change.negative { color: #f87171; }

            .footer-note {
                margin-top: 32px;
                padding-top: 24px;
                border-top: 1px solid rgba(255,255,255,0.04);
                font-size: 13px;
                color: #6b7085;
            }
            .footer-note .dev-badge {
                display: flex;
                align-items: center;
                gap: 12px;
                background: rgba(168, 85, 247, 0.08);
                padding: 6px 20px 6px 14px;
                border-radius: 40px;
                border: 1px solid rgba(168, 85, 247, 0.12);
                backdrop-filter: blur(4px);
            }
            .footer-note .dev-badge .dev-icon { font-size: 20px; }
            .footer-note .dev-badge .dev-name {
                color: #e4e7f0;
                text-decoration: none;
                font-weight: 600;
                transition: all 0.2s;
            }
            .footer-note .dev-badge .dev-name:hover {
                color: #a855f7;
                text-shadow: 0 0 20px rgba(168, 85, 247, 0.3);
            }
            .footer-note .dev-badge .social-links {
                display: flex;
                gap: 8px;
                font-size: 16px;
            }
            .footer-note .dev-badge .social-links a {
                color: #6b7085;
                text-decoration: none;
                transition: color 0.2s;
            }
            .footer-note .dev-badge .social-links a:hover { color: #a855f7; }
            .footer-note .dev-badge .divider {
                width: 1px;
                height: 20px;
                background: rgba(255,255,255,0.08);
            }
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255,255,255,0.2);
                border-radius: 50%;
                border-top-color: #fff;
                animation: spin 0.7s linear infinite;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            @media (max-width: 600px) {
                .container { padding: 28px 20px; }
                .header { flex-direction: column; align-items: stretch; }
                .search-box select { width: 100%; }
                .search-box input { min-width: 140px; width: 100%; }
                .search-box button { width: 100%; justify-content: center; }
                .prediction-grid { grid-template-columns: repeat(3, 1fr); }
                .footer-note { flex-direction: column; text-align: center; }
                .footer-note .dev-badge { justify-content: center; flex-wrap: wrap; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <div class="logo-icon">📈</div>
                    <div class="logo-text">
                        <h1>StockPredictor <span>Global</span></h1>
                    </div>
                </div>
                <div class="badge-global">Global Tracker</div>
            </div>

            <div class="status-bar" id="statusBar">
                <span class="msg" id="statusMessage">
                    <span id="statusIcon">⏳</span> 
                    <span id="statusText">Checking model...</span>
                </span>
                <button class="train-btn" id="trainBtn" style="display:none;">🚀 Train Model</button>
            </div>

            <div class="search-section">
                <div class="search-box">
                    <select id="exchangeSelect">
                        <!-- North America -->
                        <option value="US" data-currency="$">🇺🇸 United States (NYSE/NASDAQ)</option>
                        <option value="BZX" data-currency="$">🇺🇸 United States (Cboe BZX)</option>
                        <option value="PNK" data-currency="$">🇺🇸 United States (OTC Pink)</option>
                        <option value="TO" data-currency="C$">🇨🇦 Canada (Toronto)</option>
                        <option value="V" data-currency="C$">🇨🇦 Canada (TSX Venture)</option>
                        <option value="CN" data-currency="C$">🇨🇦 Canada (CSE)</option>
                        <option value="NE" data-currency="C$">🇨🇦 Canada (NEO Exchange)</option>

                        <!-- South America -->
                        <option value="SA" data-currency="R$">🇧🇷 Brazil (B3 São Paulo)</option>
                        <option value="BA" data-currency="AR$">🇦🇷 Argentina (Buenos Aires)</option>
                        <option value="SN" data-currency="CL$">🇨🇱 Chile (Santiago)</option>
                        <option value="LM" data-currency="S/">🇵🇪 Peru (Lima)</option>
                        <option value="MX" data-currency="MX$">🇲🇽 Mexico (Bolsa Mexicana)</option>

                        <!-- Europe -->
                        <option value="L" data-currency="£">🇬🇧 United Kingdom (London)</option>
                        <option value="PA" data-currency="€">🇫🇷 France (Euronext Paris)</option>
                        <option value="AS" data-currency="€">🇳🇱 Netherlands (Euronext Amsterdam)</option>
                        <option value="BR" data-currency="€">🇧🇪 Belgium (Euronext Brussels)</option>
                        <option value="LS" data-currency="€">🇵🇹 Portugal (Euronext Lisbon)</option>
                        <option value="MI" data-currency="€">🇮🇹 Italy (Borsa Italiana)</option>
                        <option value="DE" data-currency="€">🇩🇪 Germany (Xetra)</option>
                        <option value="DU" data-currency="€">🇩🇪 Germany (Düsseldorf)</option>
                        <option value="HM" data-currency="€">🇩🇪 Germany (Hamburg)</option>
                        <option value="BE" data-currency="€">🇩🇪 Germany (Berlin)</option>
                        <option value="MU" data-currency="€">🇩🇪 Germany (Munich)</option>
                        <option value="SG" data-currency="€">🇩🇪 Germany (Stuttgart)</option>
                        <option value="F" data-currency="€">🇩🇪 Germany (Frankfurt)</option>
                        <option value="VI" data-currency="€">🇦🇹 Austria (Vienna)</option>
                        <option value="SW" data-currency="Fr.">🇨🇭 Switzerland (SIX)</option>
                        <option value="HE" data-currency="€">🇫🇮 Finland (Helsinki)</option>
                        <option value="ST" data-currency="kr">🇸🇪 Sweden (Stockholm)</option>
                        <option value="CO" data-currency="kr">🇩🇰 Denmark (Copenhagen)</option>
                        <option value="OL" data-currency="kr">🇳🇴 Norway (Oslo)</option>
                        <option value="IC" data-currency="kr">🇮🇸 Iceland (Iceland Stock Exchange)</option>
                        <option value="WA" data-currency="zł">🇵🇱 Poland (Warsaw)</option>
                        <option value="PR" data-currency="Kč">🇨🇿 Czech Republic (Prague)</option>
                        <option value="AT" data-currency="€">🇬🇷 Greece (Athens)</option>
                        <option value="IR" data-currency="€">🇮🇪 Ireland (Euronext Dublin)</option>

                        <!-- Asia -->
                        <option value="NS" data-currency="₹">🇮🇳 India (NSE)</option>
                        <option value="BO" data-currency="₹">🇮🇳 India (BSE)</option>
                        <option value="T" data-currency="¥">🇯🇵 Japan (Tokyo)</option>
                        <option value="HK" data-currency="HK$">🇭🇰 Hong Kong (HKEX)</option>
                        <option value="SS" data-currency="CN¥">🇨🇳 China (Shanghai)</option>
                        <option value="SZ" data-currency="CN¥">🇨🇳 China (Shenzhen)</option>
                        <option value="KS" data-currency="₩">🇰🇷 South Korea (KOSPI)</option>
                        <option value="KQ" data-currency="₩">🇰🇷 South Korea (KOSDAQ)</option>
                        <option value="TW" data-currency="NT$">🇹🇼 Taiwan (TWSE)</option>
                        <option value="TWO" data-currency="NT$">🇹🇼 Taiwan (TPEx)</option>
                        <option value="SI" data-currency="S$">🇸🇬 Singapore (SGX)</option>
                        <option value="KL" data-currency="RM">🇲🇾 Malaysia (Bursa Malaysia)</option>
                        <option value="BK" data-currency="฿">🇹🇭 Thailand (SET)</option>
                        <option value="JK" data-currency="Rp">🇮🇩 Indonesia (IDX)</option>
                        <option value="VN" data-currency="₫">🇻🇳 Vietnam (Ho Chi Minh)</option>
                        <option value="PH" data-currency="₱">🇵🇭 Philippines (PSE)</option>

                        <!-- Oceania -->
                        <option value="AX" data-currency="A$">🇦🇺 Australia (ASX)</option>
                        <option value="NZ" data-currency="NZ$">🇳🇿 New Zealand (NZX)</option>

                        <!-- Middle East -->
                        <option value="TA" data-currency="₪">🇮🇱 Israel (Tel Aviv)</option>
                        <option value="SR" data-currency="﷼">🇸🇦 Saudi Arabia (Tadawul)</option>
                        <option value="QA" data-currency="QR">🇶🇦 Qatar (QSE)</option>
                        <option value="KW" data-currency="KD">🇰🇼 Kuwait (Kuwait Stock Exchange)</option>
                        <option value="DH" data-currency="DH">🇦🇪 UAE (Dubai)</option>
                        <option value="AD" data-currency="DH">🇦🇪 UAE (Abu Dhabi)</option>

                        <!-- Africa -->
                        <option value="JO" data-currency="R">🇿🇦 South Africa (Johannesburg)</option>
                        <option value="CA" data-currency="E£">🇪🇬 Egypt (Cairo)</option>
                    </select>
                    <input type="text" id="symbolInput" placeholder="e.g., AAPL, TSLA" value="AAPL" spellcheck="false">
                    <button id="predictBtn">🔮 Predict</button>
                </div>
                <div id="errorMessage" class="error-message"></div>
            </div>

            <div id="resultSection" class="result-section">
                <div class="stock-header">
                    <div class="stock-name"><span id="stockSymbol">AAPL</span> <small id="stockName">(US)</small></div>
                    <div class="current-price">Current: <strong id="currentPrice">$---</strong></div>
                </div>
                <div class="chart-container">
                    <canvas id="priceChart" height="200"></canvas>
                </div>
                <div class="prediction-grid" id="predictionGrid"></div>
                <div class="footer-note">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                        <span>⚠️ For educational purposes only. Not financial advice.</span>
                        <div class="dev-badge">
                            <span class="dev-icon">👨‍💻</span>
                            <a href="https://www.linkedin.com/in/ankit-arora-86a7b237/" class="dev-name">Ankit Arora</a>
                            <span class="divider"></span>
                            <div class="social-links">
                                <a href="https://github.com/ankitarora05" title="GitHub">🐙</a>
                                <a href="https://www.linkedin.com/in/ankit-arora-86a7b237/" title="LinkedIn">🔗</a>
                                <a href="mailto:05.ankitarora@gmail.com" title="Email">✉️</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // (same JavaScript as before – unchanged)
            var selectedCurrency = '';
            const exchangeSelect = document.getElementById('exchangeSelect');
            const symbolInput = document.getElementById('symbolInput');
            const predictBtn = document.getElementById('predictBtn');
            const errorDiv = document.getElementById('errorMessage');
            const resultSection = document.getElementById('resultSection');
            const stockSymbolSpan = document.getElementById('stockSymbol');
            const stockNameSpan = document.getElementById('stockName');
            const currentPriceSpan = document.getElementById('currentPrice');
            const predictionGrid = document.getElementById('predictionGrid');
            const statusText = document.getElementById('statusText');
            const statusIcon = document.getElementById('statusIcon');
            const trainBtn = document.getElementById('trainBtn');
            let chartInstance = null;
            let statusPollInterval = null;

            const placeholderMap = {
                'US': 'e.g., AAPL, TSLA, MSFT, AMZN',
                'NSE': 'e.g., RELIANCE, TCS, INFY, HDFCBANK',
                'BSE': 'e.g., RELIANCE, TCS, INFY',
                'T': 'e.g., 9984 (SoftBank), 6758 (Sony), 7203 (Toyota)',
                'HK': 'e.g., 0700 (Tencent), 9988 (Alibaba), 0005 (HSBC)',
                'SI': 'e.g., D05 (DBS), O39 (OCBC), U11 (UOB)',
                'SS': 'e.g., 600036 (CMB), 600519 (Kweichow Moutai)',
                'SZ': 'e.g., 000001 (Ping An), 000858 (Wuliangye)',
                'KS': 'e.g., 005930 (Samsung), 000660 (SK Hynix)'
            };

            exchangeSelect.addEventListener('change', function() {
                symbolInput.placeholder = placeholderMap[this.value] || 'Enter symbol';
                selectedCurrency = this[exchangeSelect.selectedIndex].getAttribute('data-currency');
            });

            function setStatus(icon, msg, isError = false) {
                statusIcon.textContent = icon;
                statusText.textContent = msg;
                statusText.style.color = isError ? '#f87171' : '#e4e7f0';
            }

            async function checkModelStatus() {
                try {
                    const resp = await fetch('/status');
                    const data = await resp.json();
                    if (data.model_exists) {
                        setStatus('✅', 'Model ready.');
                        trainBtn.style.display = 'none';
                        predictBtn.disabled = false;
                    } else {
                        setStatus('⚠️', 'No model found. Train first.');
                        trainBtn.style.display = 'inline-block';
                        predictBtn.disabled = true;
                    }
                    if (data.training_in_progress) {
                        setStatus('⏳', 'Training in progress... ' + data.training_message);
                        trainBtn.disabled = true;
                        trainBtn.textContent = '⏳ Training...';
                        predictBtn.disabled = true;
                    } else if (data.training_status === 'done') {
                        setStatus('✅', data.training_message);
                        trainBtn.style.display = 'none';
                        predictBtn.disabled = false;
                    } else if (data.training_status === 'error') {
                        setStatus('❌', data.training_message, true);
                        trainBtn.style.display = 'inline-block';
                        trainBtn.disabled = false;
                        trainBtn.textContent = '🔁 Retry Training';
                        predictBtn.disabled = true;
                    } else {
                        if (data.model_exists) {
                            setStatus('✅', 'Model ready.');
                            trainBtn.style.display = 'none';
                            predictBtn.disabled = false;
                        } else {
                            setStatus('⚠️', 'No model found. Train first.');
                            trainBtn.style.display = 'inline-block';
                            trainBtn.disabled = false;
                            trainBtn.textContent = '🚀 Train Model';
                            predictBtn.disabled = true;
                        }
                    }
                } catch (e) {
                    setStatus('❌', 'Error connecting to server.', true);
                }
            }

            async function startTraining() {
                try {
                    trainBtn.disabled = true;
                    trainBtn.textContent = '⏳ Starting...';
                    const resp = await fetch('/train', { method: 'POST' });
                    const data = await resp.json();
                    if (resp.ok) {
                        setStatus('⏳', 'Training started. This may take several minutes.');
                        if (statusPollInterval) clearInterval(statusPollInterval);
                        statusPollInterval = setInterval(checkModelStatus, 3000);
                    } else {
                        setStatus('❌', 'Error: ' + data.detail, true);
                        trainBtn.disabled = false;
                        trainBtn.textContent = '🔁 Retry Training';
                    }
                } catch (e) {
                    setStatus('❌', 'Request failed: ' + e.message, true);
                    trainBtn.disabled = false;
                    trainBtn.textContent = '🔁 Retry Training';
                }
            }

            trainBtn.addEventListener('click', startTraining);

            function clearError() {
                errorDiv.textContent = '';
                errorDiv.style.display = 'none';
            }

            function showError(msg) {
                errorDiv.textContent = msg;
                errorDiv.style.display = 'flex';
                resultSection.classList.remove('visible');
            }

            function setLoading(loading) {
                predictBtn.disabled = loading;
                predictBtn.innerHTML = loading ? '<span class="spinner"></span> Loading...' : '🔮 Predict';
            }

            function formatCurrency(value) {
                return selectedCurrency + value.toFixed(2);
            }

            function updateUI(data) {
                stockSymbolSpan.textContent = data.symbol;
                const exchange = exchangeSelect.value;
                selectedCurrency = exchangeSelect.options[exchangeSelect.selectedIndex].getAttribute('data-currency');
                const exchangeName = exchangeSelect.options[exchangeSelect.selectedIndex].text;
                stockNameSpan.textContent = `(${exchangeName})`;
                currentPriceSpan.textContent = formatCurrency(data.current_price);

                const days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'];
                const prices = [data.predictions.day_1, data.predictions.day_2, data.predictions.day_3, data.predictions.day_4, data.predictions.day_5];
                const current = data.current_price;
                let cardsHtml = '';
                days.forEach((label, i) => {
                    const price = prices[i];
                    const change = ((price - current) / current * 100).toFixed(2);
                    const sign = change >= 0 ? '+' : '';
                    const cls = change >= 0 ? 'positive' : 'negative';
                    cardsHtml += `
                        <div class="prediction-card">
                            <div class="day">${label}</div>
                            <div class="price">${formatCurrency(price)}</div>
                            <div class="change ${cls}">${sign}${change}%</div>
                        </div>
                    `;
                });
                predictionGrid.innerHTML = cardsHtml;

                const ctx = document.getElementById('priceChart').getContext('2d');
                if (chartInstance) chartInstance.destroy();

                const labels = ['Today', ...days];
                const chartData = [current, ...prices];
                const gradient = ctx.createLinearGradient(0, 0, 0, 400);
                gradient.addColorStop(0, 'rgba(168, 85, 247, 0.4)');
                gradient.addColorStop(1, 'rgba(168, 85, 247, 0.0)');

                chartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Predicted Price',
                            data: chartData,
                            borderColor: '#a855f7',
                            backgroundColor: gradient,
                            fill: true,
                            tension: 0.3,
                            pointBackgroundColor: '#a855f7',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2,
                            pointRadius: 4,
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: {
                                grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                                ticks: { color: '#8b8fa6', callback: value => selectedCurrency + value.toFixed(2) }
                            },
                            x: {
                                grid: { display: false },
                                ticks: { color: '#8b8fa6' }
                            }
                        },
                        interaction: { mode: 'index', intersect: false }
                    }
                });

                resultSection.classList.add('visible');
            }

            async function fetchPrediction(symbol, currency) {
                clearError();
                setLoading(true);
                const exchange = exchangeSelect.value;
                try {
                    const url = `/predict/${symbol}?exchange=${exchange}`;
                    const response = await fetch(url);
                    if (!response.ok) {
                        let errMsg = `Error ${response.status}`;
                        try {
                            const errData = await response.json();
                            if (errData.detail) errMsg = errData.detail;
                        } catch (_) {}
                        throw new Error(errMsg);
                    }
                    const data = await response.json();
                    if (!data.predictions || typeof data.predictions !== 'object') {
                        throw new Error('Invalid response format from server.');
                    }
                    updateUI(data, currency);
                } catch (err) {
                    showError('⚠️ ' + err.message);
                } finally {
                    setLoading(false);
                }
            }

            predictBtn.addEventListener('click', () => {
                const symbol = symbolInput.value.trim().toUpperCase();
                if (!symbol) {
                    showError('Please enter a stock symbol.');
                    return;
                }
                fetchPrediction(symbol, selectedCurrency);
            });

            symbolInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') predictBtn.click();
            });

            window.addEventListener('DOMContentLoaded', () => {
                exchangeSelect.dispatchEvent(new Event('change'));
                checkModelStatus();
                // If model exists, fetch AAPL after a short delay
                setTimeout(() => {
                    if (!predictBtn.disabled) fetchPrediction('AAPL', '$');
                }, 1500);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# =========================================================
# 5. STATUS ENDPOINT
# =========================================================
@app.get("/status")
async def status():
    model_exists = all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, FEATURES_FILE, TARGETS_FILE])
    return {
        "model_exists": model_exists,
        "training_in_progress": training_in_progress,
        "training_status": training_status,
        "training_message": training_message
    }

# =========================================================
# 6. TRAIN ENDPOINT
# =========================================================
@app.post("/train")
async def train():
    global training_in_progress
    if training_in_progress:
        raise HTTPException(status_code=400, detail="Training already in progress.")
    thread = threading.Thread(target=train_model_thread, daemon=True)
    thread.start()
    return {"message": "Training started. Check /status for progress."}

# =========================================================
# 7. PREDICTION ENDPOINT – WITH BETTER ERROR HANDLING
# =========================================================
@app.get("/predict/{symbol}")
async def predict_symbol(symbol: str, exchange: str = "US"):
    """
    Predict the next 5 closing prices for the given symbol and exchange.
    Handles NaN/Inf gracefully and returns clear HTTP errors.
    """
    global model, scaler, feature_names, target_cols

    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained. Please train the model first via the 'Train Model' button.")

    try:
        symbol = symbol.upper()
        df, used_symbol = fetch_stock_data(symbol, exchange)

        # Ensure we have a valid close price
        current_price = df['close'].iloc[-1]
        if not np.isfinite(current_price):
            # Try to find last valid price
            last_valid = df['close'].dropna()
            if len(last_valid) == 0:
                raise HTTPException(status_code=400, detail="No valid close price found for this symbol.")
            current_price = last_valid.iloc[-1]
            # Truncate the dataframe to avoid using future data (if any)
            df = df.loc[:last_valid.index[-1]]

        # Compute features
        feature_df = compute_features(df)
        feature_df_clean = feature_df.dropna()
        if len(feature_df_clean) == 0:
            raise HTTPException(status_code=400, detail="Not enough data to compute all 21-day features.")

        latest_features = feature_df_clean.iloc[-1]

        # Prepare input
        X_input = latest_features[feature_names].values.reshape(1, -1)

        # Clamp extreme values
        X_input = np.clip(X_input, -1e6, 1e6)

        if not np.isfinite(X_input).all():
            raise HTTPException(status_code=400, detail="Input features contain non-finite values.")

        # Scale
        X_scaled = scaler.transform(X_input)

        if not np.isfinite(X_scaled).all():
            raise HTTPException(
                status_code=400,
                detail="Scaled input contains NaN or infinite values. The stock price may be too extreme for the model."
            )

        # Predict returns
        predicted_returns = model.predict(X_scaled)[0]

        if not np.isfinite(predicted_returns).all():
            raise HTTPException(status_code=500, detail="Model returned non-finite predictions.")

        # Clamp returns to a reasonable range
        predicted_returns = np.clip(predicted_returns, -0.5, 0.5)

        # Cumulative prices
        price = current_price
        predictions = {}
        for i, ret in enumerate(predicted_returns):
            price = price * (1 + ret)
            predictions[f"day_{i+1}"] = round(float(price), 2)

        # Final validation
        for k, v in predictions.items():
            if not np.isfinite(v):
                raise HTTPException(status_code=500, detail=f"Prediction for {k} is not finite.")

        return {
            "symbol": used_symbol,
            "current_price": round(float(current_price), 2),
            "predictions": predictions,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"⚠️ Prediction error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")