
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
INPUT_FOLDER = "../processed"     # where clean_*.parquet are stored
OUTPUT_FOLDER = "../processed"    # KPI CSVs will also be saved here
YEAR = 2019


# ------------------------------
# LOAD CLEANED DATA
# ------------------------------
clean_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, f"clean_yellow_tripdata_{YEAR}-*.parquet")))

if not clean_files:
    raise FileNotFoundError("No cleaned parquet files found in processed/. Please run cleaning first.")

df_list = [pd.read_parquet(f) for f in clean_files]
df = pd.concat(df_list, ignore_index=True)

# ------------------------------
# PREPARE TIME COLUMNS
# ------------------------------
df['date'] = df['tpep_pickup_datetime'].dt.date
df['year'] = df['tpep_pickup_datetime'].dt.year
df['month'] = df['tpep_pickup_datetime'].dt.to_period('M')
df['week'] = df['tpep_pickup_datetime'].dt.isocalendar().week
df['dow'] = df['tpep_pickup_datetime'].dt.dayofweek   # Monday=0


# =====================================================
#   DAILY KPI
# =====================================================
kpi_daily = df.groupby('date').agg(
    trips=('VendorID', 'count'),
    duration_p50=('trip_duration', 'median'),
    duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
    speed_p50=('avg_speed', 'median')
).reset_index()

# Daily index (100 = first day)
kpi_daily['trips_index100'] = 100 * kpi_daily['trips'] / kpi_daily['trips'].iloc[0]

daily_path = os.path.join(OUTPUT_FOLDER, f"kpi_daily_{YEAR}.csv")
kpi_daily.to_csv(daily_path, index=False)


# =====================================================
#   WEEKLY KPI
# =====================================================
kpi_weekly = df.groupby('week').agg(
    trips=('VendorID', 'count'),
    duration_p50=('trip_duration', 'median'),
    duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
    speed_p50=('avg_speed', 'median')
).reset_index()

total_year_trips = kpi_weekly['trips'].sum()
kpi_weekly['pct_of_year'] = 100 * kpi_weekly['trips'] / total_year_trips

weekly_path = os.path.join(OUTPUT_FOLDER, f"kpi_weekly_{YEAR}.csv")
kpi_weekly.to_csv(weekly_path, index=False)


# =====================================================
#   MONTHLY KPI
# =====================================================
kpi_monthly = df.groupby('month').agg(
    trips=('VendorID', 'count'),
    duration_p50=('trip_duration', 'median'),
    duration_p95=('trip_duration', lambda x: x.quantile(0.95)),
    speed_p50=('avg_speed', 'median')
).reset_index()

kpi_monthly['pct_of_year'] = 100 * kpi_monthly['trips'] / kpi_monthly['trips'].sum()

monthly_path = os.path.join(OUTPUT_FOLDER, f"kpi_monthly_{YEAR}.csv")
kpi_monthly.to_csv(monthly_path, index=False)


# ------------------------------
# DONE
# ------------------------------
print("\nKPI files generated:")
print(" -", daily_path)
print(" -", weekly_path)
print(" -", monthly_path)
