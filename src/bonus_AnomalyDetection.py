import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(script_dir, '..', 'processed', 'kpi_hourly_timeseries_2019.csv')
OUTPUT_FILE = os.path.join(script_dir, '..', 'reports', 'anomalies_2019.csv')

df = pd.read_csv(INPUT_FILE)

df['revenue_per_mile'] = df['total_money'] / df['distance_sum']
df['revenue_per_mile'] = df['revenue_per_mile'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['date'] = pd.to_datetime(df['date'])
df['dow'] = df['date'].dt.dayofweek

# Z score calculation
def calculate_zscore(series):
    std = series.std()
    # Nếu dữ liệu không biến động (std=0) hoặc chỉ có 1 dòng (std=NaN) -> Z-score = 0
    if std == 0 or np.isnan(std):
        return 0
    return (series - series.mean()) / std

groups = df.groupby(['dow', 'hour'])

df['z_trips'] = groups['trips'].transform(calculate_zscore)
df['z_rev_mile'] = groups['revenue_per_mile'].transform(calculate_zscore)
df['z_speed'] = groups['speed_mean'].transform(calculate_zscore)

# DETECT ANOMALIES
THRESHOLD = 3 

# Rule 1: Volume 
df['anomaly_volume'] = np.abs(df['z_trips']) > THRESHOLD

# Rule 2: Congestion 
df['anomaly_congestion'] = df['z_speed'] < -THRESHOLD

# Rule 3: Price 
df['anomaly_price'] = df['z_rev_mile'] > THRESHOLD

# reason
def categorize_anomaly(row):
    reasons = []
    if row['anomaly_volume']:
        if row['z_trips'] > 0: reasons.append('High Demand')
        else: reasons.append('Low Demand')
    
    if row['anomaly_congestion']: reasons.append('Traffic Jam')
    if row['anomaly_price']: reasons.append('High Price')
    
    if not reasons: return 'Normal'
    return ', '.join(reasons)

df['anomaly_type'] = df.apply(categorize_anomaly, axis=1)


anomalies_df = df[df['anomaly_type'] != 'Normal'].copy()

cols_to_export = [
    'date', 'dow', 'hour',
    'trips', 'speed_mean', 'revenue_per_mile',
    'z_trips', 'z_speed', 'z_rev_mile',
    'anomaly_type'
]

final_cols = [c for c in cols_to_export if c in anomalies_df.columns]
final_output = anomalies_df[final_cols]

final_output.to_csv(OUTPUT_FILE, index=False)
