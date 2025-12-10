
import pandas as pd
import numpy as np
import glob
import os
import gc

# ------------------------------
# CONFIG
# ------------------------------
INPUT_FOLDER = "processed"     # where clean_*.parquet are stored
OUTPUT_FOLDER = "processed"    # KPI CSVs will also be saved here
YEAR = 2019


# ------------------------------
# LOAD CLEANED DATA
# ------------------------------
clean_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, f"clean_yellow_tripdata_{YEAR}-*.parquet")))

daily_chunk_pickup = []
daily_chunk_dropoff = []

weekly_chunk_pickup = []
weekly_chunk_dropoff = []

monthly_chunk_pickup = []
monthly_chunk_dropoff = []


if not clean_files:
    raise FileNotFoundError("No cleaned parquet files found in processed/. Please run cleaning first.")

for f in clean_files:
    print(f'Processing: {os.path.basename(f)}')
    df = pd.read_parquet(f)

    df['date'] = df['tpep_pickup_datetime'].dt.date
    df['year'] = df['tpep_pickup_datetime'].dt.year
    df['month'] = df['tpep_pickup_datetime'].dt.to_period('M')
    df['week'] = df['tpep_pickup_datetime'].dt.isocalendar().week
    df['dow'] = df['tpep_pickup_datetime'].dt.dayofweek   # Monday=0

    df_pickup = df.copy()
    df_pickup["LocationID"] = df_pickup["PULocationID"]

    df_dropoff = df.copy()
    df_dropoff["LocationID"] = df_dropoff["DOLocationID"]

    
    # =====================================================
    #   DAILY KPI
    # =====================================================
    daily_chunk_pickup.append(df_pickup.groupby(['date' , 'payment_type' , 'LocationID']).agg(
        trips=('VendorID', 'count'),
        duration_p50=('trip_duration', 'median'),
        duration_p95=('trip_duration', lambda x: x.quantile(0.95)), 
        speed_p50=('avg_speed', 'median'),

        # TOTAL AMOUNT
        total_money = ('total_amount' , 'sum' , 'LocationID'),

        # PASSENGER
        passenger_p50 = ( 'passenger_count' , 'median'),
        passenger_mean = ( 'passenger_count' , 'mean'),
        passenger_sum=('passenger_count', 'sum'),
        
        # DISTANCE 
        distance_sum=('trip_distance', 'sum'),
        distance_p50=('trip_distance', 'median'),
        distance_mean=('trip_distance', 'mean'),
    ))


    daily_chunk_dropoff.append(df_dropoff.groupby(['date' , 'payment_type', 'LocationID' ]).agg(
        trips=('VendorID', 'count'),
        duration_p50=('trip_duration', 'median'),
        duration_p95=('trip_duration', lambda x: x.quantile(0.95)), 
        speed_p50=('avg_speed', 'median'),

        # TOTAL AMOUNT
        total_money = ('total_amount' , 'sum'),

        # PASSENGER
        passenger_p50 = ( 'passenger_count' , 'median'),
        passenger_mean = ( 'passenger_count' , 'mean'),
        passenger_sum=('passenger_count', 'sum'),
        
        # DISTANCE 
        distance_sum=('trip_distance', 'sum'),
        distance_p50=('trip_distance', 'median'),
        distance_mean=('trip_distance', 'mean'),
    ))

    #======================================================
    # WEEKLY KPI
    #======================================================

    weekly_chunk_pickup.append(df_pickup.groupby(['week' , 'payment_type' , 'LocationID']).agg(
        trips=('VendorID', 'count'),
        duration_p50=('trip_duration', 'median'),
        duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
        speed_p50=('avg_speed', 'median'),

        # TOTAL AMOUNT
        total_money = ('total_amount' , 'sum'),

        # PASSENGER
        passenger_p50 = ( 'passenger_count' , 'median'),
        passenger_mean = ( 'passenger_count' , 'mean'),
        passenger_sum=('passenger_count', 'sum'),
        
        # DISTANCE
        distance_p50=('trip_distance', 'median'),
        distance_mean=('trip_distance', 'mean'),
        distance_sum=('trip_distance', 'sum'),

        
    ))

    weekly_chunk_dropoff.append(df_dropoff.groupby(['week' , 'payment_type' , 'LocationID']).agg(
        trips=('VendorID', 'count'),
        duration_p50=('trip_duration', 'median'),
        duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
        speed_p50=('avg_speed', 'median'),

        # TOTAL AMOUNT
        total_money = ('total_amount' , 'sum'),

        # PASSENGER
        passenger_p50 = ( 'passenger_count' , 'median'),
        passenger_mean = ( 'passenger_count' , 'mean'),
        passenger_sum=('passenger_count', 'sum'),
        
        # DISTANCE
        distance_p50=('trip_distance', 'median'),
        distance_mean=('trip_distance', 'mean'),
        distance_sum=('trip_distance', 'sum'),

        
    ))

    # =====================================================
    #   MONTHLY KPI
    # =====================================================
    monthly_chunk_pickup.append(df_pickup.groupby(['month' , 'payment_type' , 'LocationID']).agg(
        trips=('VendorID', 'count'),
        duration_p50=('trip_duration', 'median'),
        duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
        speed_p50=('avg_speed', 'median'),


        # TOTAL AMOUNT
        total_money = ('total_amount' , 'sum'),

        # PASSENGER
        passenger_p50 = ( 'passenger_count' , 'median'),
        passenger_mean = ( 'passenger_count' , 'mean'),
        passenger_sum=('passenger_count', 'sum'),

        # DISTANCE
        distance_sum=('trip_distance', 'sum'),
        distance_p50=('trip_distance', 'median'),
        distance_mean=('trip_distance', 'mean'),
    ))

    monthly_chunk_dropoff.append(df_dropoff.groupby(['month' , 'payment_type' , 'LocationID']).agg(
        trips=('VendorID', 'count'),
        duration_p50=('trip_duration', 'median'),
        duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
        speed_p50=('avg_speed', 'median'),


        # TOTAL AMOUNT
        total_money = ('total_amount' , 'sum'),

        # PASSENGER
        passenger_p50 = ( 'passenger_count' , 'median'),
        passenger_mean = ( 'passenger_count' , 'mean'),
        passenger_sum=('passenger_count', 'sum'),

        # DISTANCE
        distance_sum=('trip_distance', 'sum'),
        distance_p50=('trip_distance', 'median'),
        distance_mean=('trip_distance', 'mean'),
    ))


    del df
    gc.collect()


#-------------------------------
#   CHUNK MERGE
#-------------------------------

kpi_daily_pickup = pd.concat(daily_chunk_pickup).reset_index()
kpi_weekly_pickup = pd.concat(weekly_chunk_pickup).reset_index()
kpi_monthly_pickup = pd.concat(monthly_chunk_pickup).reset_index()

kpi_daily_dropoff = pd.concat(daily_chunk_dropoff).reset_index()
kpi_weekly_dropoff = pd.concat(weekly_chunk_dropoff).reset_index()
kpi_monthly_dropoff = pd.concat(monthly_chunk_dropoff).reset_index()

# =====================================================
#   COMPUTE % OF YEAR TRIPS
# =====================================================
kpi_weekly_pickup["pct_of_year"] = 100 * kpi_weekly_pickup["trips"] / kpi_weekly_pickup["trips"].sum()
kpi_weekly_dropoff["pct_of_year"] = 100 * kpi_weekly_dropoff["trips"] / kpi_weekly_dropoff["trips"].sum()
kpi_monthly_pickup["pct_of_year"] = 100 * kpi_monthly_pickup["trips"] / kpi_monthly_pickup["trips"].sum()
kpi_monthly_dropoff["pct_of_year"] = 100 * kpi_monthly_dropoff["trips"] / kpi_monthly_dropoff["trips"].sum()


# =====================================================
#   SAVE OUTPUTS
# =====================================================
daily_pu_path  = os.path.join(OUTPUT_FOLDER, f"kpi_daily_pickup{YEAR}.csv")
daily_do_path  = os.path.join(OUTPUT_FOLDER, f"kpi_daily_dropoff{YEAR}.csv")
weekly_pu_path = os.path.join(OUTPUT_FOLDER, f"kpi_weekly_pickup_{YEAR}.csv")
weekly_do_path = os.path.join(OUTPUT_FOLDER, f"kpi_weekly_dropoff_{YEAR}.csv")
monthly_pu_path = os.path.join(OUTPUT_FOLDER, f"kpi_monthly_pickup{YEAR}.csv")
monthly_do_path = os.path.join(OUTPUT_FOLDER, f"kpi_monthly_dropoff{YEAR}.csv")

kpi_daily_pickup.to_csv(daily_pu_path, index=False)
kpi_daily_dropoff.to_csv(daily_do_path, index=False)
kpi_weekly_pickup.to_csv(weekly_pu_path, index=False)
kpi_weekly_dropoff.to_csv(weekly_do_path, index=False)
kpi_monthly_pickup.to_csv(monthly_pu_path, index=False)
kpi_monthly_dropoff.to_csv(monthly_do_path, index=False)

print("\nKPI files generated:")
print(" -", daily_pu_path)
print(" -", daily_do_path)
print(" -", weekly_pu_path)
print(" -", weekly_do_path)
print(" -", monthly_pu_path)
print(" -", monthly_do_path)
