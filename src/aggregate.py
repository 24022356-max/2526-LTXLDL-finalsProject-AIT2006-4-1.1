
import pandas as pd
import numpy as np
import glob
import os
import gc

# ------------------------------
# CONFIG
# ------------------------------
INPUT_FOLDER = 'processed'     # where clean_*.parquet are stored
OUTPUT_FOLDER = 'processed'    # KPI CSVs will also be saved here
YEAR = 2019


# ------------------------------
# LOAD CLEANED DATA
# ------------------------------
clean_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, f'clean_yellow_tripdata_{YEAR}-*.parquet')))

chunks = {
    'hourly': [],
    'daily': [],
    'weekly': [],
    'monthly': [],
    'monthly_pickup': [],
    'monthly_dropoff': [],
    'monthly_payment_type': []
}


if not clean_files:
    raise FileNotFoundError('No cleaned parquet files found in processed/. Please run cleaning first.')

# ------------------------------
# AGGREGATE LOGIC
# ------------------------------
def apply_agg(df, group_cols):
    return df.groupby(group_cols).agg(
        trips=('VendorID', 'count'),
        duration_p50=('trip_duration', 'median'),
        duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
        duration_mean=('trip_duration', 'mean'),
        speed_p50=('avg_speed', 'median'),
        speed_mean=('avg_speed', 'mean'),
        total_money=('total_amount', 'sum'),
        passenger_mean=('passenger_count', 'mean'),
        passenger_sum=('passenger_count', 'sum'),
        distance_sum=('trip_distance', 'sum'),
        distance_p50=('trip_distance', 'median'),
        distance_mean=('trip_distance', 'mean'),
    )

for f in clean_files:
    print(f'Processing: {os.path.basename(f)}')
    df = pd.read_parquet(f)

    df['date'] = df['tpep_pickup_datetime'].dt.date
    df['year'] = df['tpep_pickup_datetime'].dt.year
    df['month'] = df['tpep_pickup_datetime'].dt.to_period('M')
    df['week_start'] = df['tpep_pickup_datetime'].dt.to_period('W').dt.start_time
    df['dow']  = df['tpep_pickup_datetime'].dt.dayofweek # 0=Monday, 6=Sunday
    df['hour'] = df['tpep_pickup_datetime'].dt.hour # 0 to 23
    
    # =====================================================
    #  KPI
    # =====================================================
    chunks['hourly'].append(apply_agg(df, ['dow', 'hour']))
    chunks['daily'].append(apply_agg(df, ['date']))
    chunks['weekly'].append(apply_agg(df, ['week_start']))
    chunks['monthly'].append(apply_agg(df, ['month']))
    
    chunks['monthly_pickup'].append(apply_agg(df, ['month', 'PULocationID']))
    chunks['monthly_dropoff'].append(apply_agg(df, ['month', 'DOLocationID']))
    chunks['monthly_payment_type'].append(apply_agg(df, ['month', 'payment_type']))

    del df
    gc.collect()


#-------------------------------
#   CHUNK MERGE
#-------------------------------

kpi = {
    'hourly': pd.DataFrame(),
    'daily': pd.DataFrame(),
    'weekly': pd.DataFrame(),
    'monthly': pd.DataFrame(),
    'monthly_pickup': pd.DataFrame(),
    'monthly_dropoff': pd.DataFrame(),
    'monthly_payment_type': pd.DataFrame()
}

for key in kpi:
    kpi[key] = pd.concat(chunks[key]).reset_index()

# ------------------------------
# Fix duplicate rows in weekly
# ------------------------------

kpi['weekly'] = kpi['weekly'].groupby('week_start').agg(
    trips=('trips', 'sum'),
    total_money=('total_money', 'sum'),
    passenger_sum=('passenger_sum', 'sum'),
    distance_sum=('distance_sum', 'sum'),
    duration_p50=('duration_p50', 'mean'), 
    duration_p95=('duration_p95', 'mean'),
    duration_mean=('duration_mean', 'mean'),
    speed_p50=('speed_p50', 'mean'),
    speed_mean=('speed_mean', 'mean'),
    distance_p50=('distance_p50', 'mean'),
).reset_index()

kpi['weekly'] = kpi['weekly'].sort_values('week_start')

# ------------------------------
# Merge Hourly
# ------------------------------
kpi['hourly'] = kpi['hourly'].groupby(['dow', 'hour']).agg(
    trips=('trips', 'sum'),
    total_money=('total_money', 'sum'),
    passenger_sum=('passenger_sum', 'sum'),
    distance_sum=('distance_sum', 'sum'),
    duration_p50=('duration_p50', 'mean'), 
    duration_p95=('duration_p95', 'mean'),
    duration_mean=('duration_mean', 'mean'),
    speed_p50=('speed_p50', 'mean'),
    speed_mean=('speed_mean', 'mean'),
    distance_p50=('distance_p50', 'mean'),
).reset_index()

dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
kpi['hourly']['day'] = kpi['hourly']['dow'].map(dow_map)

# =====================================================
#   COMPUTE PERCENTAGE AND SAVE OUTPUTS
# =====================================================
output_path = {
    'hourly': f'kpi_hourly_{YEAR}.csv',
    'daily': f'kpi_daily_{YEAR}.csv',
    'weekly': f'kpi_weekly_{YEAR}.csv',
    'monthly': f'kpi_monthly_{YEAR}.csv',
    'monthly_pickup': f'kpi_monthly_pickup_{YEAR}.csv',
    'monthly_dropoff': f'kpi_monthly_dropoff_{YEAR}.csv',
    'monthly_payment_type': f'kpi_monthly_payment_type_{YEAR}.csv'
}

total_trips_year = kpi['monthly']['trips'].sum()
total_money_year = kpi['monthly']['total_money'].sum()

for key, filename in output_path.items():
    kpi[key]['trip_pct'] = (kpi[key]['trips'] / total_trips_year) * 100
    kpi[key]['money_pct'] = (kpi[key]['total_money'] / total_money_year) * 100

    kpi[key].to_csv(os.path.join(OUTPUT_FOLDER, filename), index=False)
    print(f"Saved: {filename}")