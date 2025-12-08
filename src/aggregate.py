
import pandas as pd
import numpy as np
import glob
import os
import gc
import pandas as pd
import numpy as np
import glob
import os

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

daily_chunk = []
weekly_chunk = []
monthly_chunk = []


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

    # =====================================================
    #   DAILY KPI
    # =====================================================
    daily_chunk.append(df.groupby('date').agg(
    trips=('VendorID', 'count'),
    duration_p50=('trip_duration', 'median'),
    duration_p95=('trip_duration', lambda x: x.quantile(0.95)), 
    speed_p50=('avg_speed', 'median'))
    )

    weekly_chunk.append(df.groupby('week').agg(
        trips=('VendorID', 'count'),
        duration_p50=('trip_duration', 'median'),
        duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
        speed_p50=('avg_speed', 'median')
    ))

    # =====================================================
    #   MONTHLY KPI
    # =====================================================
    monthly_chunk.append(df.groupby('month').agg(
    trips=('VendorID', 'count'),
    duration_p50=('trip_duration', 'median'),
    duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
    speed_p50=('avg_speed', 'median')
    ))


    del df
    gc.collect()


#-------------------------------
#   CHUNK MERGE
#-------------------------------

kpi_daily = pd.concat(daily_chunk).reset_index()
kpi_weekly = pd.concat(weekly_chunk).groupby("week").sum().reset_index()
kpi_monthly = pd.concat(monthly_chunk).groupby("month").sum().reset_index()

# =====================================================
#   COMPUTE % OF YEAR TRIPS
# =====================================================
kpi_weekly["pct_of_year"] = 100 * kpi_weekly["trips"] / kpi_weekly["trips"].sum()
kpi_monthly["pct_of_year"] = 100 * kpi_monthly["trips"] / kpi_monthly["trips"].sum()


# =====================================================
#   SAVE OUTPUTS
# =====================================================
daily_path = os.path.join(OUTPUT_FOLDER, f"kpi_daily_{YEAR}.csv")
weekly_path = os.path.join(OUTPUT_FOLDER, f"kpi_weekly_{YEAR}.csv")
monthly_path = os.path.join(OUTPUT_FOLDER, f"kpi_monthly_{YEAR}.csv")

kpi_daily.to_csv(daily_path, index=False)
kpi_weekly.to_csv(weekly_path, index=False)
kpi_monthly.to_csv(monthly_path, index=False)

print("\nKPI files generated:")
print(" -", daily_path)
print(" -", weekly_path)
print(" -", monthly_path)