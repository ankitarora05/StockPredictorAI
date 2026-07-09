# train_stock_model.py
import yfinance as yf
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

print("Downloading stock data...")
ticker = "JPM"  # Change to 'TSLA', 'MSFT', etc.
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = "2015-01-01"

df = yf.download(ticker, start=start_date, end=end_date, progress=False)

# === BULLETPROOF COLUMN CLEANUP ===
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
print(f"Downloaded {len(df)} trading days. Columns: {df.columns.tolist()}")

# === FEATURE ENGINEERING ===
def create_features(data):
    df_feat = data.copy()
    
    # Lagged prices & volume
    for lag in [1, 5, 10, 21]:
        df_feat[f'close_lag_{lag}'] = df_feat['close'].shift(lag)
        df_feat[f'volume_lag_{lag}'] = df_feat['volume'].shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 21]:
        df_feat[f'close_ma_{window}'] = df_feat['close'].rolling(window).mean()
        df_feat[f'close_std_{window}'] = df_feat['close'].rolling(window).std()
        df_feat[f'volume_ma_{window}'] = df_feat['volume'].rolling(window).mean()
    
    # Ratios & Returns
    df_feat['high_low_ratio'] = df_feat['high'] / df_feat['low']
    df_feat['close_open_ratio'] = df_feat['close'] / df_feat['open']
    df_feat['daily_return'] = df_feat['close'].pct_change()
    df_feat['return_lag_1'] = df_feat['daily_return'].shift(1)
    
    # *** NEW: Targets for Day+1, Day+2, Day+3, Day+4, Day+5 (as percentages) ***
    for i in range(1, 6):
        df_feat[f'target_{i}d'] = (df_feat['close'].shift(-i) / df_feat['close']) - 1
    
    return df_feat

df_feat = create_features(df)
df_feat = df_feat.dropna()
print(f"Shape after feature engineering: {df_feat.shape}")

# === TRAIN / TEST SPLIT ===
features = [col for col in df_feat.columns if not col.startswith('target_')]
X = df_feat[features]

# Create a 2D y array with 5 columns (one for each day)
target_cols = ['target_1d', 'target_2d', 'target_3d', 'target_4d', 'target_5d']
y = df_feat[target_cols]  # Shape: (n_samples, 5)

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# === SCALE & TRAIN (Multi-output Random Forest) ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RandomForest handles multi-output automatically when y is 2D
model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Evaluate (calculate average MAE across all 5 targets)
y_pred = model.predict(X_test_scaled)
mae_per_day = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
avg_mae = np.mean(mae_per_day)
print(f"✅ Test MAE per day (returns): {mae_per_day}")
print(f"✅ Average MAE (returns): {avg_mae:.4f}")

# Save MAE for reference in predictions (optional, but good for confidence intervals)
joblib.dump(model, 'stock_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(features, 'features.joblib')
joblib.dump(target_cols, 'target_cols.joblib')  # Save this so prediction script knows order

print(f"✅ Model saved! Expects {len(features)} features and outputs {len(target_cols)} targets.")