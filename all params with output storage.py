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

# Create DataFrames for each analysis

# Hourly Transactions
hourly_transactions = data.groupby(data['timestamp'].dt.hour).size().reset_index(name='transaction_count')
hourly_transactions.columns = ['hour', 'transaction_count']

# Range of Hours
start_hour = 8
end_hour = 12
range_hourly_count = data[(data['timestamp'].dt.hour >= start_hour) & (data['timestamp'].dt.hour < end_hour)].shape[0]
range_hourly_df = pd.DataFrame({'hour_range': [f'{start_hour}-{end_hour}'], 'transaction_count': [range_hourly_count]})

# Daily Transactions
daily_transactions = data.groupby(data['timestamp'].dt.date).size().reset_index(name='transaction_count')
daily_transactions.columns = ['date', 'transaction_count']

# Range of Days
start_date = pd.Timestamp('2024-09-01')
end_date = pd.Timestamp('2024-09-03')
range_daily_count = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)].shape[0]
range_daily_df = pd.DataFrame({'date_range': [f'{start_date.date()} to {end_date.date()}'], 'transaction_count': [range_daily_count]})

# Weekly Transactions
weekly_transactions = data.groupby(data['timestamp'].dt.to_period('W')).size().reset_index(name='transaction_count')
weekly_transactions.columns = ['week', 'transaction_count']

# Range of Weeks
start_week = pd.Timestamp('2024-08-26')  # Example start date for a week
end_week = pd.Timestamp('2024-09-01')    # Example end date for a week
range_weekly_count = data[(data['timestamp'] >= start_week) & (data['timestamp'] <= end_week)].shape[0]
range_weekly_df = pd.DataFrame({'week_range': [f'{start_week.date()} to {end_week.date()}'], 'transaction_count': [range_weekly_count]})

# Weekends vs. Weekdays
weekend_transactions = data[data['timestamp'].dt.dayofweek >= 5].shape[0]
weekday_transactions = data[data['timestamp'].dt.dayofweek < 5].shape[0]
weekends_weekdays_df = pd.DataFrame({
    'type': ['weekends', 'weekdays'],
    'transaction_count': [weekend_transactions, weekday_transactions]
})

# Monthly Transactions
monthly_transactions = data.groupby(data['timestamp'].dt.to_period('M')).size().reset_index(name='transaction_count')
monthly_transactions.columns = ['month', 'transaction_count']

# Range of Months
start_month = pd.Timestamp('2024-09-01')
end_month = pd.Timestamp('2024-11-30')
range_monthly_count = data[(data['timestamp'] >= start_month) & (data['timestamp'] <= end_month)].shape[0]
range_monthly_df = pd.DataFrame({'month_range': [f'{start_month.date()} to {end_month.date()}'], 'transaction_count': [range_monthly_count]})

# Yearly Transactions
yearly_transactions = data.groupby(data['timestamp'].dt.year).size().reset_index(name='transaction_count')
yearly_transactions.columns = ['year', 'transaction_count']

# Indian Holidays
holidays_df = pd.DataFrame({
    'holiday_date': ['2024-08-15', '2024-10-02']
})
holidays_df['holiday_date'] = pd.to_datetime(holidays_df['holiday_date'])
holiday_transactions = data[data['timestamp'].dt.date.isin(holidays_df['holiday_date'].dt.date)].shape[0]
holidays_summary_df = pd.DataFrame({'type': ['holidays'], 'transaction_count': [holiday_transactions]})

# Indian Festivals
festivals_df = pd.DataFrame({
    'festival_date': ['2024-10-24', '2024-12-25']
})
festivals_df['festival_date'] = pd.to_datetime(festivals_df['festival_date'])
festival_transactions = data[data['timestamp'].dt.date.isin(festivals_df['festival_date'].dt.date)].shape[0]
festivals_summary_df = pd.DataFrame({'type': ['festivals'], 'transaction_count': [festival_transactions]})

# Combine all DataFrames
result_df = pd.concat([
    hourly_transactions,
    range_hourly_df,
    daily_transactions,
    range_daily_df,
    weekly_transactions,
    range_weekly_df,
    weekends_weekdays_df,
    monthly_transactions,
    range_monthly_df,
    yearly_transactions,
    holidays_summary_df,
    festivals_summary_df
], ignore_index=True)

# Save to CSV
result_df.to_csv('transaction_analysis_summary.csv', index=False)

print("Data saved to 'transaction_analysis_summary.csv'")
