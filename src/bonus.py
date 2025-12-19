import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.holiday import USFederalHolidayCalendar
import os
import warnings
warnings.filterwarnings('ignore')

# ================ CONFIG ================#
DATA_PATH = 'processed'
REPORTS_PATH = 'reports'
YEAR = 2019
FORECAST_DAYS = 60

os.makedirs(REPORTS_PATH, exist_ok=True)

# =============== LOAD DATA =============#
def load_kpi_daily(year=YEAR, folder=DATA_PATH):
    file_path = os.path.join(folder, f'kpi_daily_{year}.csv')
    print(f"Loading: {file_path}")
    
    df = pd.read_csv(
        file_path,
        usecols=['date', 'trips'],
        parse_dates=['date'],
        index_col='date'
    )
    df = df.asfreq('D')
    print(f"Loaded: {len(df)} days")
    return df

data = load_kpi_daily()

# ============= BASELINE FORECAST ============#
print("\nCalculating Baseline...")
last_year = data['trips'].iloc[-365:]
baseline_future = np.tile(last_year.values, (FORECAST_DAYS // 365) + 1)[:FORECAST_DAYS]

# ========== LINEAR REGRESSION FORECAST =======#
print("\nCalculating Linear Regression...")
def get_calendar_features(date):
    day_num = date.dayofweek
    month_num = date.month
    
    feats = {'is_holiday': 0} # Default, updated later
    
    # One-Hot Day of Week
    for d in range(7):
        feats[f'day_{d}'] = 1 if day_num == d else 0
        
    # One-Hot Month
    for m in range(1, 13):
        feats[f'month_{m}'] = 1 if month_num == m else 0
        
    return feats

# Fill missing data to prevent crashes
data = data.asfreq('D').fillna(method='ffill')
start_date = data.index[0] # Still needed for the 'days_from_start' calculation below

train_rows = []
for date in data.index:
    row = get_calendar_features(date)
    
    # Lag 7: If we are in the first week (Jan 1-7), we "borrow" data from Dec 25-31
    days_from_start = (date - start_date).days
    
    if days_from_start < 7:
        lag_value = data['trips'].iloc[-(7 - days_from_start)]
        row['lag_7'] = lag_value
    else:
        row['lag_7'] = data.loc[date - pd.Timedelta(days=7), 'trips']
    
    row['target'] = data.loc[date, 'trips']
    train_rows.append(row)

train_df = pd.DataFrame(train_rows)

# Holidays
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=data.index.min(), end=data.index.max())
train_df['is_holiday'] = data.index.isin(holidays).astype(int)

# Train
X_train = train_df.drop(columns=['target'])
y_train = train_df['target']

model = LinearRegression()
model.fit(X_train, y_train)

# Forecast
history = data['trips'].tolist()
future_predictions = []
current_date = data.index[-1] + pd.Timedelta(days=1)

for i in range(FORECAST_DAYS):
    feat_dict = get_calendar_features(current_date)
    feat_dict['is_holiday'] = 1 if current_date in cal.holidays() else 0
    feat_dict['lag_7'] = history[-7]
    
    this_day_X = pd.DataFrame([feat_dict])
    this_day_X = this_day_X[X_train.columns] 
    
    pred = model.predict(this_day_X)[0]
    
    future_predictions.append(pred)
    history.append(pred)
    current_date += pd.Timedelta(days=1)

# Save results
future_dates=pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS)
results = pd.DataFrame({
    'Date': future_dates, 
    'Forecast': future_predictions
})

# =============== ARIMA FORECAST =============#
print("\nTraining ARIMA...")

def create_exog(dates):
    exog = pd.DataFrame(index=dates)
    # Holidays are the ONLY external thing we need
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=dates.min(), end=dates.max())
    exog['is_holiday'] = exog.index.isin(holidays).astype(int)
    return exog
y_train_log = np.log(data['trips'])
exog_train = create_exog(data.index)
exog_future = create_exog(future_dates)

arima_model = SARIMAX(
    y_train_log,
    exog=exog_train,
    order=(2, 1, 1),              
    seasonal_order=(1, 0, 1, 7)
)

fitted = arima_model.fit(disp=False)

arima_log = fitted.forecast(steps=FORECAST_DAYS, exog=exog_future)
arima_future = np.exp(arima_log + 0.25 * fitted.mse)

# ============= SAVE RESULTS =============#
print("\nSaving predictions...")

results = pd.DataFrame({
    'Date': future_dates,
    'Baseline': baseline_future,
    'Linear_Reg': future_predictions,
    'ARIMA': arima_future
})

# Save to reports folder
results.to_csv(f'{REPORTS_PATH}/forecast_results_{YEAR + 1}.csv', index=False)

historical = data[['trips']].copy()
historical.columns = ['Historical']
historical.index.name = 'date'
historical.to_csv(f'{REPORTS_PATH}/historical_data_{YEAR}.csv')

print("\nDone!")