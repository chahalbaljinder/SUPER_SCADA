import pandas as pd
import holidays

# Sample DataFrame
data = {
    'timestamp': ['2024-01-01 08:15:00', '2024-01-01 09:00:00', '2024-01-02 08:30:00', '2024-01-05 10:00:00', '2024-01-05 15:00:00'],
    'transaction_id': [1, 2, 3, 4, 5],
    'transaction_sequence': [1, 2, 3, 4, 5]
}

df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert to datetime
df.set_index('timestamp', inplace=True)

# 1. Number of Transactions in an Hour
transactions_per_hour = df.resample('H').size()
print("Transactions per hour:")
print(transactions_per_hour)

# 2. Number of Transactions in a Range of Hours
start_hour = '08:00:00'
end_hour = '12:00:00'
filtered_df_hours = df.between_time(start_time=start_hour, end_time=end_hour)
transactions_in_range_of_hours = filtered_df_hours.resample('D').size()
print(f"\nTransactions between {start_hour} and {end_hour} each day:")
print(transactions_in_range_of_hours)

# 3. Number of Transactions in a Day
transactions_per_day = df.resample('D').size()
print("\nTransactions per day:")
print(transactions_per_day)

# 4. Number of Transactions in a Range of Days
start_date = '2024-01-01'
end_date = '2024-01-05'
filtered_df_days = df[start_date:end_date]
transactions_in_range_of_days = filtered_df_days.resample('D').size()
print(f"\nTransactions from {start_date} to {end_date}:")
print(transactions_in_range_of_days)

# 5. Number of Transactions in a Week
transactions_per_week = df.resample('W').size()
print("\nTransactions per week:")
print(transactions_per_week)

# 6. Number of Transactions in a Range of Weeks
start_week = '2024-01-01'
end_week = '2024-01-31'
filtered_df_weeks = df[start_week:end_week]
transactions_in_range_of_weeks = filtered_df_weeks.resample('W').size()
print(f"\nTransactions from {start_week} to {end_week}:")
print(transactions_in_range_of_weeks)

# 7. Number of Transactions on Weekends vs. Weekdays
df['weekday'] = df.index.day_name()
transactions_by_weekday = df.groupby('weekday').size()
print("\nTransactions by weekday:")
print(transactions_by_weekday)

weekend_days = ['Saturday', 'Sunday']
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x in weekend_days else 0)
transactions_weekends_vs_weekdays = df.groupby('is_weekend').size()
print("Transactions on weekends vs. weekdays:")
print(transactions_weekends_vs_weekdays)

# 8. Number of Transactions in a Month
transactions_per_month = df.resample('M').size()
print("\nTransactions per month:")
print(transactions_per_month)

# 9. Number of Transactions in a Range of Months
start_month = '2024-01'
end_month = '2024-03'
filtered_df_months = df[start_month:end_month]
transactions_in_range_of_months = filtered_df_months.resample('M').size()
print(f"\nTransactions from {start_month} to {end_month}:")
print(transactions_in_range_of_months)

# 10. Number of Transactions on Indian Holidays
indian_holidays = holidays.India(years=2024)
holiday_dates = pd.DataFrame(list(indian_holidays.items()), columns=['date', 'holiday'])
holiday_dates['date'] = pd.to_datetime(holiday_dates['date'])

# Check transactions on holidays
transactions_on_holidays = df[df.index.normalize().isin(holiday_dates['date'])]
transactions_on_holidays_count = transactions_on_holidays.shape[0]
print(f"\nNumber of transactions on Indian holidays: {transactions_on_holidays_count}")
