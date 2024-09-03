import pandas as pd

# Sample Data
data = {
    'timestamp': ['2024-09-01 10:00:00', '2024-09-02 12:00:00', '2024-09-03 15:00:00',
                  '2024-09-04 09:00:00', '2024-09-05 14:00:00', '2024-09-06 20:00:00',
                  '2024-09-07 18:00:00', '2024-09-08 08:00:00'],
    'transaction_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'transaction_sequence': [101, 102, 103, 104, 105, 106, 107, 108]
}

df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert to datetime

# Add a 'week_number' column
df['week_number'] = df['timestamp'].dt.isocalendar().week

# Add a 'day_of_week' column (0 = Monday, 6 = Sunday)
df['day_of_week'] = df['timestamp'].dt.dayofweek

# 1. Number of transactions per week
transactions_per_week = df.groupby('week_number').size()
print('Transactions per week:')
print(transactions_per_week)

# 2. Number of transactions in a range of weeks
start_week = 35
end_week = 36
transactions_in_range = df[(df['week_number'] >= start_week) & (df['week_number'] <= end_week)].shape[0]
print(f'\nTransactions between week {start_week} and week {end_week}:', transactions_in_range)

# 3. Number of transactions on weekends
weekend_transactions = df[df['day_of_week'] >= 5].shape[0]
print('\nNumber of weekend transactions:', weekend_transactions)

# 4. Number of transactions on weekdays
weekday_transactions = df[df['day_of_week'] < 5].shape[0]
print('\nNumber of weekday transactions:', weekday_transactions)
