import pandas as pd
import numpy as np
import glob
import os
import gc
from tqdm import tqdm

input_folder = '../raw'
output_folder = '../processed'
report_folder = '../reports'

# List to collect QA results for every month
all_qa_stats = []

def process_month(file_path):
    filename = os.path.basename(file_path)
    month_str = filename.split('_')[-1].replace('.parquet', '') 
    
    # Load Data
    df = pd.read_parquet(file_path, engine='pyarrow')

    # Fix Types
    float_cols = [
        'VendorID', 'passenger_count', 'RatecodeID', 'payment_type', 
        'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 
        'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'airport_fee'
    ]
    # Only convert cols that exist
    exist_float_cols = [c for c in float_cols if c in df.columns]
    df[exist_float_cols] = df[exist_float_cols].astype('float32')
    
    # Fix Dates
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Calc Derived Metrics
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Avoid division by zero for speed
    df['avg_speed'] = np.where(
        df['trip_duration'] > 0, 
        df['trip_distance'] / (df['trip_duration'] / 60), 
        0
    )

    # Fill NaNs
    values_to_fill = {
        'passenger_count': 1, 'RatecodeID': 1, 'store_and_fwd_flag': 'N',
        'congestion_surcharge': 0, 'airport_fee': 0, 'avg_speed': 0
    }
    df.fillna(values_to_fill, inplace=True)

    # Apply Rules & Collect Stats
    df, qa_stats = apply_qa_rules(df, month_str)
    
    # Add stats to our global list
    all_qa_stats.append(qa_stats)

    # Return ONLY valid rows and drop duplicates
    valid_df = df[df['is_valid_trip'] & ~df.duplicated()].copy()
    
    # Drop the temporary columns we created
    cols_to_drop = [c for c in valid_df.columns if c.startswith('qa_')] + ['is_valid_trip']
    valid_df.drop(columns=cols_to_drop, inplace=True)
    
    return valid_df

def apply_qa_rules(df, month_str):
    n_total = len(df)
    current_month_dt = pd.to_datetime(month_str)
    next_month_dt = current_month_dt + pd.DateOffset(months=1)

    # --- RULES ---
    df['qa_dropoff_after_pickup'] = df['trip_duration'] > 0
    df['qa_timedate'] = (df['tpep_pickup_datetime'] >= current_month_dt) & (df['tpep_pickup_datetime'] < next_month_dt)
    
    # Duration: < 10 hours (600 mins)
    df['qa_duration'] = (df['trip_duration'] > 0) & (df['trip_duration'] < 600) 
    
    # Distance: > 0 miles OR 0 miles with > $0 fare
    df['qa_distance'] = ((df['trip_distance'] > 0) | ((df['trip_distance'] == 0) & (df['total_amount'] > 0)))
    
    # Speed: < 70 mph
    df['qa_speed'] = (df['avg_speed'] >= 0) & (df['avg_speed'] <= 70) 

    # Payment Type: 0 to 6
    df['qa_payment_type'] = (df['payment_type'] >= 0) & (df['payment_type'] <= 6)

    # Total Amount: > $0 and < $1000 AND matches sum of components
    df['qa_total_amount'] = (df['total_amount'] > 0) & (df['total_amount'] < 1000)
    
    # Tip: >= $0 and <= total_amount
    df['qa_tip_amount'] = (df['tip_amount'] >= 0) & (df['tip_amount'] <= df['total_amount'])
    
    # Non-Negative Charges
    df['qa_non_neg_amount'] = (
        (df['fare_amount'] >= 0) & (df['extra'] >= 0) & (df['mta_tax'] >= 0) & 
        (df['improvement_surcharge'] >= 0) & (df['tolls_amount'] >= 0) & 
        (df['congestion_surcharge'] >= 0) & (df['airport_fee'] >= 0)
    )

    # Location IDs: 1 to 265 but PULocationID must be in NYC
    df['qa_locationID'] = (
        (df['PULocationID'] > 0) & (df['PULocationID'] <= 263) & 
        (df['DOLocationID'] > 0) & (df['DOLocationID'] <= 265)
    )
    
    # RatecodeID: 1 to 6
    df['qa_ratecodeID'] = (df['RatecodeID'] > 0) & (df['RatecodeID'] <= 6)
    
    # VendorID: 1, 2, 6, 7
    df['qa_vendorID'] = df['VendorID'].isin([1, 2, 6, 7]) 

    # Passenger Count: 0 to 9
    df['qa_passenger_count'] = (df['passenger_count'] >= 0) & (df['passenger_count'] <= 9)

    # --- VALIDATION ---
    qa_cols = [col for col in df.columns if col.startswith('qa_')]
    df['is_valid_trip'] = df[qa_cols].all(axis=1)

    # --- STATS COLLECTION ---
    stats = {'month': month_str, 'total_rows': n_total}
    
    # Calculate fail percentage for each rule
    for col in qa_cols:
        n_fail = (~df[col]).sum()
        stats[f'{col}_fail_pct'] = round((n_fail / n_total) * 100, 2)
    
    n_invalid = (~(df['is_valid_trip'] & ~df.duplicated())).sum()
    stats['total_dropped_count'] = n_invalid
    stats['total_dropped_pct'] = round((n_invalid / n_total) * 100, 2)
    
    return df, stats

# --- MAIN ---
raw_files = sorted(glob.glob(os.path.join(input_folder, 'yellow_tripdata_2019-*.parquet')))

for file in tqdm(raw_files, desc="Processing Files"):
    try:
        cleaned_df = process_month(file)
        
        if cleaned_df is not None and not cleaned_df.empty:
            output_name = os.path.basename(file)
            save_path = os.path.join(output_folder, f"clean_{output_name}")
            cleaned_df.to_parquet(save_path, index=False)
        else:
            print(f"\n{file}: No data")

        del cleaned_df
        gc.collect()

    except Exception as e:
        print(f"\nFailed on {file}: {e}")

# --- SAVE REPORT ---
if all_qa_stats:
    report_df = pd.DataFrame(all_qa_stats)
    report_df.to_csv(os.path.join(report_folder, 'qa_summary.csv'), index=False)
    print(report_df[['month', 'total_dropped_pct']])