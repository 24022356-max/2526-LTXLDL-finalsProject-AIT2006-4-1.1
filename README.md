# 2526-LTXLDL-finalsProject-AIT2006-4-1.1 - NYC Taxi Data Analysis (2019)

This is the finals project of group AIT2006-4-1.1.

## Description

This project analyzes 12 months of Yellow Taxi trip records to identify seasonal trends, outlier fares, and drop-off patterns. The data was cleaned using Python and visualized to show revenue fluctuations over the year.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Installation

## Project Structure

- `src/`: Contains the analysis notebooks and cleaning scripts in Python.
- `raw/`: Folder for raw Parquet files (Not uploaded to GitHub due to size).
- `processed/`: Folder where cleaned data is saved (Not uploaded to GitHub due to size).
- `reports/`: Generated QA reports and summary CSVs.
- `figures/`: Generated graphs for clearer visualization.
- `requirements.txt`: Python dependencies.

## Usage

1. Download the raw data from [NYC TLC Website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) into the `raw/` folder.
2. Run the cleaning script:
   ```bash
   python src/clean_data.py
   ```

## Data Cleaning Steps

We processed ~80 million rows. Data was filtered based on the following rules:
- **Dates:** Retained trips only within the year 2019.
- **Passengers:** Kept trips with 0-9 passengers (0 allowed for delivery/app errors).
- **Fares:** Removed negative fares and outliers > $1000.
- **Distance/Duration:** Removed negative distances and trips > 10 hours.
- **Speed:** Removed trips with impossible speeds (> 70 mph).
- **Missing Values:** Filled NaNs with 0 for monetary columns.

**QA Result:** Approximately 2% of raw data was dropped as invalid on average.

## Findings

- **Seasonality:** Revenue peaked in **Decemeber** and dipped in **December**.
- **Outliers:** Found valid trips costing over $500, verified as negotiated fares.