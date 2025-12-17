import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ================ CONFIG ================#
DATA_PATH = 'processed'
YEAR = 2019
FORECAST_DAYS = 30
REPORTS_PATH = 'reports'



# =============== LOAD DATA =============#
def load_kpi_daily(year=YEAR, folder=DATA_PATH):
    file_path = os.path.join(folder, f'kpi_daily_{year}.csv')
    print(f" Loading: {file_path}")
    
    df = pd.read_csv(
        file_path,
        usecols=['date', 'trips'],
        parse_dates=['date'],
        index_col='date'
    )
    df = df.asfreq('D')
    print(f" Loaded: {len(df)} days")
    return df

### DATA LOADING ###
data = load_kpi_daily()

# ============= BASELINE FORECAST ============#
last_week = data['trips'].iloc[-7:]
baseline_future = np.tile(last_week.values, (FORECAST_DAYS //7) + 1 )[:FORECAST_DAYS]



# ============ LINEAR REGRESSION FORCAST ==========#
features = pd.DataFrame(index = data.index)
features['dayofweek'] = features.index.dayofweek.astype('int8')
features['month']  = features.inndex.month.astype('int8')
features['lag_7'] = data['trips'].shift(7)
features['lag_30'] = data['trips'].shift(30)
features['rolling_7'] = data['trips'].shift(1).rolling(7).mean()

X_train = features.dropna()
Y_train = data.loc[X_train.index , 'trips']


#  Trainn model

lr_model = LinearRegression(n_jobs= -1)
lr_model.fit(X_train , Y_train)

# future models
future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS)
future_features = pd.DataFrame(index=future_dates)
future_features['dayofweek'] = future_features.index.dayofweek.astype('int8')
future_features['month'] = future_features.index.month.astype('int8')
future_features['lag_7'] = data['trips'].iloc[-7:].values
future_features['lag_30'] = data['trips'].iloc[-30:].values
future_features['rolling_7'] = data['trips'].iloc[-7:].mean()

# Predict
lr_future = lr_model.predict(future_features)
lr_future = np.maximum(lr_future, 0).astype('int32')



# =============== ARIMA FORECAST =============#
print("\Training ARIMA...")
arima_train = data['trips'].iloc[-200:]  # Dùng 200 điểm cuối
arima_model = SARIMAX(arima_train, order=(2, 1, 2), low_memory=True)
arima_fitted = arima_model.fit(disp=False, low_memory=True)
arima_future = arima_fitted.forecast(steps=FORECAST_DAYS)
arima_future = np.maximum(arima_future, 0).astype('int32')



# Chuẩn bị dictionary predictions
predictions = {
    'Baseline': baseline_future,
    'Linear Regression': lr_future,
    'ARIMA': arima_future
}

# ============= SAVE SUMMARY =============#
results = pd.DataFrame({
    'Date': future_dates,
    'Baseline': baseline_future,
    'Linear_Reg': lr_future,
    'ARIMA': arima_future
})

results.to_csv(f'{REPORTS_PATH}/forecast_results_{YEAR}.csv', index=False)

historical = data[['trips']].copy()
historical.columns = ['Historical']
historical.index.name = 'date'
historical.to_csv(f'{REPORTS_PATH}/historical_data_{YEAR}.csv')