import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import warnings
warnings.filterwarnings('ignore')

# ================ CONFIG ================#
DATA_PATH = 'processed'
REPORTS_PATH = 'reports'
YEAR = 2019
FORECAST_DAYS = 30

os.makedirs(REPORTS_PATH, exist_ok=True)

# =============== LOAD DATA =============#
def load_kpi_daily(year=YEAR, folder=DATA_PATH):
    file_path = os.path.join(folder, f'kpi_daily_{year}.csv')
    print(f"Loading: {file_path}")
    
    df = pd.read_csv(
        file_path,
        usecols=['date', 'trips'],
        parse_dates=['date'],
        index_col='date'
    )
    df = df.asfreq('D')
    print(f"Loaded: {len(df)} days")
    return df

data = load_kpi_daily()

# ============= BASELINE FORECAST ============#
print("\nCalculating Baseline...")
last_week = data['trips'].iloc[-7:]
baseline_future = np.tile(last_week.values, (FORECAST_DAYS // 7) + 1)[:FORECAST_DAYS]

# ========== LINEAR REGRESSION FORECAST =======#
print("\nCalculating Linear Regression...")
features = pd.DataFrame(index=data.index)
features['dayofweek'] = features.index.dayofweek.astype('int8')
features['month'] = features.index.month.astype('int8')
features['lag_7'] = data['trips'].shift(7)
features['lag_30'] = data['trips'].shift(30)
features['rolling_7'] = data['trips'].shift(1).rolling(7).mean()

X_train = features.dropna()
y_train = data.loc[X_train.index, 'trips']

lr_model = LinearRegression(n_jobs=-1)
lr_model.fit(X_train, y_train)

# ============== CREATE FUTURE FEATURES =============#
future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS)
future_features = pd.DataFrame(index=future_dates)
future_features['dayofweek'] = future_features.index.dayofweek.astype('int8')
future_features['month'] = future_features.index.month.astype('int8')

# ✅ FIX LENGTH MISMATCH - Tile lag patterns
lag_7_values = data['trips'].iloc[-7:].values
future_features['lag_7'] = np.tile(lag_7_values, (FORECAST_DAYS // 7) + 1)[:FORECAST_DAYS]

lag_30_values = data['trips'].iloc[-30:].values
future_features['lag_30'] = np.tile(lag_30_values, (FORECAST_DAYS // 30) + 1)[:FORECAST_DAYS]

# Rolling mean is a single value, pandas will broadcast it
future_features['rolling_7'] = data['trips'].iloc[-7:].mean()

# Predict
lr_future = lr_model.predict(future_features)
lr_future = np.maximum(lr_future, 0).astype('int32')

# =============== ARIMA FORECAST =============#
print("\nTraining ARIMA...")
arima_train = data['trips'].iloc[-200:]
arima_model = SARIMAX(arima_train, order=(2, 1, 2), low_memory=True)
arima_fitted = arima_model.fit(disp=False, low_memory=True)
arima_future = arima_fitted.forecast(steps=FORECAST_DAYS)
arima_future = np.maximum(arima_future, 0).astype('int32')

# ============= SAVE RESULTS =============#
print("\nSaving predictions...")

results = pd.DataFrame({
    'Date': future_dates,
    'Baseline': baseline_future,
    'Linear_Reg': lr_future,
    'ARIMA': arima_future
})

# Save to reports folder
results.to_csv(f'{REPORTS_PATH}/forecast_results_{YEAR}.csv', index=False)

historical = data[['trips']].copy()
historical.columns = ['Historical']
historical.index.name = 'date'
historical.to_csv(f'{REPORTS_PATH}/historical_data_{YEAR}.csv')

print(f"Saved: {REPORTS_PATH}/forecast_results_{YEAR}.csv")
print(f"Saved: {REPORTS_PATH}/historical_data_{YEAR}.csv")
print("\nDone!")