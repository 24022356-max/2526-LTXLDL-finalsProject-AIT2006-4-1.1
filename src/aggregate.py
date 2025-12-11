
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
def apply_agg(df):
    return df.agg(
        trips=('VendorID', 'count'),
        duration_p50=('trip_duration', 'median'),
        duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
        speed_p50=('avg_speed', 'median'),
        total_money=('total_amount', 'sum'),
        passenger_p50=('passenger_count', 'median'),
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

    base_col = ['date', 'month', 'year', 'week_start',
                'VendorID', 'payment_type', 'PULocationID', 'DOLocationID',
                'passenger_count', 'trip_distance', 'total_amount', 
                'avg_speed', 'trip_duration']
    df_base = df[base_col].copy()
    
    # =====================================================
    #  KPI
    # =====================================================
    chunks['daily'].append(apply_agg(df.groupby(['date'])))
    chunks['weekly'].append(apply_agg(df.groupby('week_start')))
    chunks['monthly'].append(apply_agg(df.groupby(['month'])))
    
    chunks['monthly_pickup'].append(apply_agg(df.groupby(['month', 'PULocationID'])))
    chunks['monthly_dropoff'].append(apply_agg(df.groupby(['month', 'DOLocationID'])))
    chunks['monthly_payment_type'].append(apply_agg(df.groupby(['month', 'payment_type'])))

    del df, df_base
    gc.collect()


#-------------------------------
#   CHUNK MERGE
#-------------------------------

kpi_daily = pd.concat(chunks['daily']).reset_index()
kpi_monthly = pd.concat(chunks['monthly']).reset_index()

kpi_monthly_pickup = pd.concat(chunks['monthly_pickup']).reset_index()
kpi_monthly_dropoff = pd.concat(chunks['monthly_dropoff']).reset_index()
kpi_monthly_payment_type = pd.concat(chunks['monthly_payment_type']).reset_index()

# ------------------------------
# Fix duplicate rows in weekly
# ------------------------------

kpi_weekly = (pd.concat(chunks['weekly']).reset_index()).groupby('week_start').agg(
    trips=('trips', 'sum'),
    total_money=('total_money', 'sum'),
    passenger_sum=('passenger_sum', 'sum'),
    distance_sum=('distance_sum', 'sum'),
    duration_p50=('duration_p50', 'mean'), 
    duration_p95=('duration_p95', 'mean'),
    speed_p50=('speed_p50', 'mean'),
    passenger_p50=('passenger_p50', 'mean'),
    distance_p50=('distance_p50', 'mean'),
).reset_index()
kpi_weekly['passenger_mean'] = kpi_weekly['passenger_sum'] / kpi_weekly['trips']
kpi_weekly['distance_mean'] = kpi_weekly['distance_sum'] / kpi_weekly['trips']
kpi_weekly = kpi_weekly.sort_values('week_start')

# =====================================================
#   COMPUTE PERCENTAGE AND SAVE OUTPUTS
# =====================================================
output_path = {
    'daily': f'kpi_daily_{YEAR}.csv',
    'weekly': f'kpi_weekly_{YEAR}.csv',
    'monthly': f'kpi_monthly_{YEAR}.csv',
    'monthly_pickup': f'kpi_monthly_pickup_{YEAR}.csv',
    'monthly_dropoff': f'kpi_monthly_dropoff_{YEAR}.csv',
    'monthly_payment_type': f'kpi_monthly_payment_type_{YEAR}.csv'
}

temp_monthly = pd.concat(chunks['monthly'])
total_trips_year = temp_monthly['trips'].sum()
total_money_year = temp_monthly['total_money'].sum()

for key, filename in output_path.items():
    df_final = pd.concat(chunks[key]).reset_index()
    
    # Calculate Percentages
    df_final['trip_pct'] = (df_final['trips'] / total_trips_year) * 100
    df_final['money_pct'] = (df_final['total_money'] / total_money_year) * 100

    df_final.to_csv(os.path.join(OUTPUT_FOLDER, filename), index=False)
    print(f"Saved: {filename}")