# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 1. LOAD TRAINED ARTIFACTS
# =========================================================
model = joblib.load('stock_model.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('features.joblib')

try:
    target_cols = joblib.load('target_cols.joblib')
except FileNotFoundError:
    target_cols = ['target_1d', 'target_2d', 'target_3d', 'target_4d', 'target_5d']

app = FastAPI(
    title="Stock Predictor AI",
    description="Predict the next 5 closing prices for any stock using a trained Random Forest model.",
    version="2.0"
)

# =========================================================
# 2. HELPER FUNCTIONS (EXACTLY MATCH TRAINING)
# =========================================================

def clean_columns(df):
    """Handles MultiIndex columns from yfinance."""
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
        raise KeyError(f"Could not find required columns in {df.columns}. Found: {col_map}")
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
    return df_feat

def fetch_stock_data(symbol):
    end_date = datetime.today().strftime('%Y-%m-%d')
    days_back = 30
    df = None
    while days_back <= 60:
        start_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if len(df) >= 21:
            break
        days_back += 5
    if df is None or len(df) < 21:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough historical data for '{symbol}'. Only found {len(df) if df is not None else 0} trading days."
        )
    return clean_columns(df)

# =========================================================
# 3. API ENDPOINTS
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the beautiful frontend UI."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Predictor AI – Next 5 Days Forecast</title>
        <meta name="description" content="AI-powered stock price prediction for the next 5 trading days. Enter any stock symbol and get instant forecasts.">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
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
                -webkit-backdrop-filter: blur(20px);
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
            .badge-ai {
                background: rgba(168, 85, 247, 0.15);
                border: 1px solid rgba(168, 85, 247, 0.25);
                padding: 6px 18px;
                border-radius: 40px;
                font-size: 13px;
                font-weight: 600;
                color: #c084fc;
                letter-spacing: 0.3px;
                display: inline-flex;
                align-items: center;
                gap: 6px;
            }
            .badge-ai::before {
                content: "⚡";
                font-size: 16px;
            }
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
            }
            .search-box input {
                flex: 1;
                min-width: 200px;
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
            .search-box input::placeholder {
                color: #6b7085;
            }
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
            .search-box button:hover {
                transform: scale(1.02);
                box-shadow: 0 12px 32px rgba(106, 90, 249, 0.4);
            }
            .search-box button:disabled {
                opacity: 0.5;
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
            .result-section.visible {
                display: block;
            }
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
                font-size: 16px;
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
                font-size: 13px;
                color: #6b7085;
                text-align: center;
                border-top: 1px solid rgba(255,255,255,0.04);
                padding-top: 24px;
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
                .search-box input { min-width: 140px; }
                .search-box button { width: 100%; justify-content: center; }
                .prediction-grid { grid-template-columns: repeat(3, 1fr); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <div class="logo-icon">📈</div>
                    <div class="logo-text">
                        <h1>StockPredictor <span>AI</span></h1>
                    </div>
                </div>
                <div class="badge-ai">AI Powered</div>
            </div>

            <div class="search-section">
                <div class="search-box">
                    <input type="text" id="symbolInput" placeholder="Enter stock symbol (e.g., AAPL, TSLA, MSFT)" value="AAPL" spellcheck="false">
                    <button id="predictBtn">🔮 Predict</button>
                </div>
                <div id="errorMessage" class="error-message"></div>
            </div>

            <div id="resultSection" class="result-section">
                <div class="stock-header">
                    <div class="stock-name"><span id="stockSymbol">AAPL</span> <small id="stockName">Apple Inc.</small></div>
                    <div class="current-price">Current: <strong id="currentPrice">$---</strong></div>
                </div>

                <div class="chart-container">
                    <canvas id="priceChart" height="200"></canvas>
                </div>

                <div class="prediction-grid" id="predictionGrid">
                    <!-- cards will be inserted by JS -->
                </div>

                <div class="footer-note">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                        <span>⚠️ Predictions are based on a Random Forest model trained on historical data. Not financial advice.</span>
                        <div style="display: flex; align-items: center; gap: 8px; background: rgba(168, 85, 247, 0.1); padding: 4px 16px 4px 12px; border-radius: 40px; border: 1px solid rgba(168, 85, 247, 0.15);">
                            <span style="font-size: 18px;">👨‍💻</span>
                            <span style="font-size: 13px; color: #c084fc; font-weight: 500;">Built by <a href="https://www.linkedin.com/in/ankit-arora-86a7b237/" style="color: #e4e7f0; text-decoration: none; font-weight: 600; transition: color 0.2s;" onmouseover="this.style.color='#a855f7'" onmouseout="this.style.color='#e4e7f0'">Ankit Arora</a></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const symbolInput = document.getElementById('symbolInput');
            const predictBtn = document.getElementById('predictBtn');
            const errorDiv = document.getElementById('errorMessage');
            const resultSection = document.getElementById('resultSection');
            const stockSymbolSpan = document.getElementById('stockSymbol');
            const stockNameSpan = document.getElementById('stockName');
            const currentPriceSpan = document.getElementById('currentPrice');
            const predictionGrid = document.getElementById('predictionGrid');
            let chartInstance = null;

            // Helper to clear error
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
                return '$' + value.toFixed(2);
            }

            function updateUI(data) {
                // data = { symbol, current_price, predictions: { day_1, day_2, ... } }
                stockSymbolSpan.textContent = data.symbol;
                // We don't have company name from yfinance, so we just show the symbol.
                stockNameSpan.textContent = data.symbol;
                currentPriceSpan.textContent = formatCurrency(data.current_price);

                // Build prediction cards
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

                // Build chart
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
                        plugins: {
                            legend: { display: false },
                        },
                        scales: {
                            y: {
                                grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                                ticks: { color: '#8b8fa6', callback: value => '$' + value.toFixed(2) }
                            },
                            x: {
                                grid: { display: false },
                                ticks: { color: '#8b8fa6' }
                            }
                        },
                        interaction: {
                            mode: 'index',
                            intersect: false
                        }
                    }
                });

                resultSection.classList.add('visible');
            }

            async function fetchPrediction(symbol) {
                clearError();
                setLoading(true);
                try {
                    const response = await fetch(`/predict/${symbol}`);
                    if (!response.ok) {
                        let errMsg = `Error ${response.status}`;
                        try {
                            const errData = await response.json();
                            if (errData.detail) errMsg = errData.detail;
                        } catch (_) {}
                        throw new Error(errMsg);
                    }
                    const data = await response.json();
                    // Validate data
                    if (!data.predictions || typeof data.predictions !== 'object') {
                        throw new Error('Invalid response format from server.');
                    }
                    updateUI(data);
                } catch (err) {
                    showError('⚠️ ' + err.message);
                } finally {
                    setLoading(false);
                }
            }

            // Event listeners
            predictBtn.addEventListener('click', () => {
                const symbol = symbolInput.value.trim().toUpperCase();
                if (!symbol) {
                    showError('Please enter a stock symbol.');
                    return;
                }
                fetchPrediction(symbol);
            });

            symbolInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') predictBtn.click();
            });

            // Load default on page load
            window.addEventListener('DOMContentLoaded', () => {
                fetchPrediction('AAPL');
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/predict/{symbol}")
async def predict_symbol(symbol: str):
    """
    Predict the closing price for the next 5 trading days for any stock symbol.
    """
    try:
        symbol = symbol.upper()
        df = fetch_stock_data(symbol)
        feature_df = compute_features(df)
        feature_df_clean = feature_df.dropna()
        if len(feature_df_clean) == 0:
            raise HTTPException(status_code=400, detail="Not enough data to compute all 21-day features.")
        latest_features = feature_df_clean.iloc[-1]
        current_price = df['close'].iloc[-1]

        X_input = latest_features[feature_names].values.reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        predicted_returns = model.predict(X_scaled)[0]  # shape (5,)

        price = current_price
        predictions = {}
        for i, ret in enumerate(predicted_returns):
            price = price * (1 + ret)
            predictions[f"day_{i+1}"] = round(float(price), 2)

        return {
            "symbol": symbol,
            "current_price": round(float(current_price), 2),
            "predictions": predictions,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")