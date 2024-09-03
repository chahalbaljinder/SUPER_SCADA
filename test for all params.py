import pandas as pd
import numpy as np

# Sample DataFrame
data = pd.DataFrame({
    'timestamp': ['2024-09-01 08:30:00', '2024-09-01 09:00:00', '2024-09-02 10:00:00'],
    'transaction_id': [1, 2, 3],
    'transaction_sequence': [1, 2, 3]
})

# Convert timestamp column to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Hourly Transactions
hourly_transactions = data.groupby(data['timestamp'].dt.hour).size()
print("Transactions by hour:\n", hourly_transactions)

# Range of Hours
start_hour = 8
end_hour = 12
range_hourly_transactions = data[(data['timestamp'].dt.hour >= start_hour) & (data['timestamp'].dt.hour < end_hour)]
range_hourly_count = range_hourly_transactions.shape[0]
print(f"Transactions from hour {start_hour} to {end_hour}:\n", range_hourly_count)

# Daily Transactions
daily_transactions = data.groupby(data['timestamp'].dt.date).size()
print("Transactions by day:\n", daily_transactions)

# Range of Days
start_date = pd.Timestamp('2024-09-01')
end_date = pd.Timestamp('2024-09-03')
range_daily_transactions = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
range_daily_count = range_daily_transactions.shape[0]
print(f"Transactions from {start_date.date()} to {end_date.date()}:\n", range_daily_count)

# Weekly Transactions
weekly_transactions = data.groupby(data['timestamp'].dt.to_period('W')).size()
print("Transactions by week:\n", weekly_transactions)

# Range of Weeks
start_week = pd.Timestamp('2024-08-26')  # Example start date for a week
end_week = pd.Timestamp('2024-09-01')    # Example end date for a week
range_weekly_transactions = data[(data['timestamp'] >= start_week) & (data['timestamp'] <= end_week)]
range_weekly_count = range_weekly_transactions.shape[0]
print(f"Transactions from week starting {start_week.date()} to {end_week.date()}:\n", range_weekly_count)

# Weekends vs. Weekdays
data['is_weekend'] = data['timestamp'].dt.dayofweek >= 5
weekend_transactions = data[data['is_weekend']].shape[0]
weekday_transactions = data[~data['is_weekend']].shape[0]
print("Weekend transactions:\n", weekend_transactions)
print("Weekday transactions:\n", weekday_transactions)

# Monthly Transactions
monthly_transactions = data.groupby(data['timestamp'].dt.to_period('M')).size()
print("Transactions by month:\n", monthly_transactions)

# Range of Months
start_month = pd.Timestamp('2024-09-01')
end_month = pd.Timestamp('2024-11-30')
range_monthly_transactions = data[(data['timestamp'] >= start_month) & (data['timestamp'] <= end_month)]
range_monthly_count = range_monthly_transactions.shape[0]
print(f"Transactions from {start_month.date()} to {end_month.date()}:\n", range_monthly_count)

# Yearly Transactions
yearly_transactions = data.groupby(data['timestamp'].dt.year).size()
print("Transactions by year:\n", yearly_transactions)

# Indian Holidays (Sample DataFrame)
holidays_df = pd.DataFrame({
    'holiday_date': ['2024-08-15', '2024-10-02']
})
holidays_df['holiday_date'] = pd.to_datetime(holidays_df['holiday_date'])

# Transactions on Holidays
data['is_holiday'] = data['timestamp'].dt.date.isin(holidays_df['holiday_date'].dt.date)
holiday_transactions = data[data['is_holiday']].shape[0]
print("Transactions on holidays:\n", holiday_transactions)

# Indian Festivals (Sample DataFrame, Adjust as Needed)
festivals_df = pd.DataFrame({
    'festival_date': ['2024-10-24', '2024-12-25']
})
festivals_df['festival_date'] = pd.to_datetime(festivals_df['festival_date'])

# Transactions on Festivals
data['is_festival'] = data['timestamp'].dt.date.isin(festivals_df['festival_date'].dt.date)
festival_transactions = data[data['is_festival']].shape[0]
print("Transactions on festivals:\n", festival_transactions)
