
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

daily_chunk = []
weekly_chunk = []
monthly_chunk = []

# payment_type, PULocationID and DOLocationID KPIs for monthly only
monthly_chunk_pickup = []
monthly_chunk_dropoff = []
monthly_chunk_payment_type = []


if not clean_files:
    raise FileNotFoundError('No cleaned parquet files found in processed/. Please run cleaning first.')

for f in clean_files:
    print(f'Processing: {os.path.basename(f)}')
    df = pd.read_parquet(f)

    df['date'] = df['tpep_pickup_datetime'].dt.date
    df['year'] = df['tpep_pickup_datetime'].dt.year
    df['month'] = df['tpep_pickup_datetime'].dt.to_period('M')
    df['week'] = df['tpep_pickup_datetime'].dt.isocalendar().week
    df['dow'] = df['tpep_pickup_datetime'].dt.dayofweek   # Monday=0

    base_col = ['date', 'dow', 'week', 'month', 'year', 
                'VendorID', 'payment_type', 'PULocationID', 'DOLocationID',
                'passenger_count', 'trip_distance', 'total_amount', 
                'avg_speed', 'trip_duration']
    df_base = df[base_col].copy()
    
    # =====================================================
    #   DAILY KPI
    # =====================================================
    daily_chunk.append(
    df_base.groupby(['date']).agg(
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
    )


    #======================================================
    # WEEKLY KPI
    #======================================================

    weekly_chunk.append(
    df_base.groupby(['week']).agg(
        trips=('VendorID', 'count'),
        duration_p50=('trip_duration', 'median'),
        duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
        speed_p50=('avg_speed', 'median'),

        total_money=('total_amount', 'sum'),

        passenger_p50=('passenger_count', 'median'),
        passenger_mean=('passenger_count', 'mean'),
        passenger_sum=('passenger_count', 'sum'),

        distance_p50=('trip_distance', 'median'),
        distance_mean=('trip_distance', 'mean'),
        distance_sum=('trip_distance', 'sum'),
    )
    )

    # =====================================================
    #   MONTHLY KPI
    # =====================================================
    monthly_chunk.append(
    df_base.groupby(['month']).agg(
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
    )

    monthly_chunk_pickup.append(
    df_base.groupby(['month', 'PULocationID']).agg(
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
    )


    monthly_chunk_dropoff.append(
    df_base.groupby(['month', 'DOLocationID']).agg(
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
    )

    monthly_chunk_payment_type.append(
    df_base.groupby(['month', 'payment_type']).agg(
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
    )



    del df, df_base
    gc.collect()


#-------------------------------
#   CHUNK MERGE
#-------------------------------

kpi_daily = pd.concat(daily_chunk).reset_index()
kpi_weekly = pd.concat(weekly_chunk).reset_index()
kpi_monthly = pd.concat(monthly_chunk).reset_index()

kpi_monthly_pickup = pd.concat(monthly_chunk_pickup).reset_index()
kpi_monthly_dropoff = pd.concat(monthly_chunk_dropoff).reset_index()
kpi_monthly_payment_type = pd.concat(monthly_chunk_payment_type).reset_index()

# =====================================================
#   COMPUTE % OF YEAR TRIPS AND MONEY
# =====================================================
total_trips = kpi_monthly['trips'].sum()

kpi_daily['trip_pct'] = 100 * kpi_daily['trips'] / total_trips
kpi_weekly['trip_pct'] = 100 * kpi_weekly['trips'] / total_trips
kpi_monthly['trip_pct'] = 100 * kpi_monthly['trips'] / total_trips

kpi_monthly_payment_type['trip_pct'] = 100 * kpi_monthly_payment_type['trips'] / total_trips
kpi_monthly_pickup['trip_pct'] = 100 * kpi_monthly_pickup['trips'] / total_trips
kpi_monthly_dropoff['trip_pct'] = 100 * kpi_monthly_dropoff['trips'] / total_trips

total_money = kpi_monthly['total_money'].sum()

kpi_daily['money_pct'] = 100 * kpi_daily['total_money'] / total_money
kpi_weekly['money_pct'] = 100 * kpi_weekly['total_money'] / total_money
kpi_monthly['money_pct'] = 100 * kpi_monthly['total_money'] / total_money

kpi_monthly_payment_type['money_pct'] = 100 * kpi_monthly_payment_type['total_money'] / total_money
kpi_monthly_pickup['money_pct'] = 100 * kpi_monthly_pickup['total_money'] / total_money
kpi_monthly_dropoff['money_pct'] = 100 * kpi_monthly_dropoff['total_money'] / total_money


# =====================================================
#   SAVE OUTPUTS
# =====================================================
daily_path  = os.path.join(OUTPUT_FOLDER, f'kpi_daily_{YEAR}.csv')
weekly_path = os.path.join(OUTPUT_FOLDER, f'kpi_weekly_{YEAR}.csv')
monthly_path = os.path.join(OUTPUT_FOLDER, f'kpi_monthly_{YEAR}.csv')

monthly_pt_path = os.path.join(OUTPUT_FOLDER, f'kpi_monthly_payment_type_{YEAR}.csv')
monthly_pu_path = os.path.join(OUTPUT_FOLDER, f'kpi_monthly_pickup_{YEAR}.csv')
monthly_do_path = os.path.join(OUTPUT_FOLDER, f'kpi_monthly_dropoff_{YEAR}.csv')

kpi_daily.to_csv(daily_path, index=False)
kpi_weekly.to_csv(weekly_path, index=False)
kpi_monthly.to_csv(monthly_path, index=False)

kpi_monthly_pickup.to_csv(monthly_pu_path, index=False)
kpi_monthly_dropoff.to_csv(monthly_do_path, index=False)
kpi_monthly_payment_type.to_csv(monthly_pt_path, index=False)
