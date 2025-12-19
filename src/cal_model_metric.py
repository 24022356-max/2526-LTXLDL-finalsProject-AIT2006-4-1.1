import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# ============= CALCULATE MODEL PERFOMANCE METRICS (RMSE, MAE & MAPE) =============#

# Run after bonus.py
# Only run this if you have data from 2020 in raw/
# The pre-calculated metrics is in reports/

REAL_FOLDER = 'raw'
FORECAST_FOLDER = 'reports'
OUTPUT_FOLDER = 'reports'
MODELS = ['Baseline', 'Linear_Reg', 'ARIMA']

# Only predict the first 2 months
YEAR = 2020
MONTHS = [1, 2] # Jan, Feb

# Function for loading real data
def load_real_data(data_dir, year, months):
    daily_counts = []
    
    for month in months:
        file_name = f'yellow_tripdata_{year}-{month:02d}.parquet'
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_parquet(file_path, engine='pyarrow')

        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        mask = (df["tpep_pickup_datetime"].dt.year == year) & \
                (df["tpep_pickup_datetime"].dt.month == month)
        df = df.loc[mask].copy()
        df['Date'] = df['tpep_pickup_datetime'].dt.normalize()

        daily_agg = df.groupby('Date').size().reset_index(name='Real_Data')
        daily_counts.append(daily_agg)

    full_df = pd.concat(daily_counts, ignore_index=True)
    full_df.sort_values('Date', inplace=True)
    return full_df

# Function for MAPE
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Load real and prediction data
df_truth = load_real_data(REAL_FOLDER, YEAR, MONTHS)
df_forecast = pd.read_csv(os.path.join(FORECAST_FOLDER, f'forecast_results_{YEAR}.csv'))

# Merge
date_col = 'Date' if 'Date' in df_forecast.columns else 'date'
df_forecast[date_col] = pd.to_datetime(df_forecast[date_col])
eval_df = pd.merge(df_forecast, df_truth, left_on=date_col, right_on='Date', how='inner')
# Save to forecast results csv
eval_df.to_csv(f'{OUTPUT_FOLDER}/forecast_results_{YEAR}.csv', index=False)

# Calculate
metrics = {}
for model in MODELS:
    y_true = eval_df['Real_Data']
    y_pred = eval_df[model]
    
    metrics[model] = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE (%)': calculate_mape(y_true, y_pred)
    }

# Save
metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv('reports/model_performance_metrics.csv')