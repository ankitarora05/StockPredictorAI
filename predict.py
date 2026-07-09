# predict.py
import yfinance as yf
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# 1. Load artifacts
model = joblib.load('stock_model.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('features.joblib')
target_cols = joblib.load('target_cols.joblib')  # ['target_1d', ..., 'target_5d']

# 2. Fetch data - dynamically get at least 21 trading days
ticker = "JPM"  # Must match the ticker you trained on
end_date = datetime.today().strftime('%Y-%m-%d')

days_back = 30
df = None
while days_back <= 60:
    start_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if len(df) >= 21:
        break
    days_back += 5

print(f"Fetched {len(df)} trading days from {start_date} to {end_date}.")

# === COLUMN CLEANUP ===
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
    df = df[list(col_map.values())]
    df.columns = list(col_map.keys())
    return df

df = clean_columns(df)

# 3. Compute features (EXACT copy from training)
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

feature_df = compute_features(df)

# 4. Drop NaNs and get the latest complete row
feature_df_clean = feature_df.dropna()
if len(feature_df_clean) == 0:
    print("❌ Error: Not enough data to compute all 21-day features.")
    exit()

latest_complete = feature_df_clean.iloc[-1]
last_actual_close = df['close'].iloc[-1]  # Today's closing price

X_input = latest_complete[feature_names].values.reshape(1, -1)

# 5. Scale and Predict (Outputs an array of 5 returns)
X_scaled = scaler.transform(X_input)
predicted_returns = model.predict(X_scaled)[0]  # Shape: (5,)

# 6. Calculate cumulative prices for each of the next 5 days
print(f"\n📈 {ticker} - Forecast for the Next 5 Trading Days:")
print(f"   Today's Close: ${last_actual_close:.2f}\n")
print(f"   {'Day':<10} {'Return':<15} {'Predicted Close':<20}")
print(f"   {'-'*10} {'-'*15} {'-'*20}")

cumulative_price = last_actual_close
for i, ret in enumerate(predicted_returns):
    day_num = i + 1
    # Cumulative growth: price_today * (1 + ret1) * (1 + ret2) ...
    cumulative_price = cumulative_price * (1 + ret)
    target_date = (datetime.today() + timedelta(days=day_num * 2)).strftime('%Y-%m-%d')  # Approx date
    print(f"   Day +{day_num:<5} {ret * 100:>+8.2f}%    ${cumulative_price:>12.2f}")

print(f"\n   Target dates are estimates (weekends excluded).")