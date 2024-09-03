import pandas as pd

# Example data
data = {
    'timestamp': ['2023-09-01 12:34:56', '2023-09-01 13:15:22', '2023-09-02 10:23:45'],
    'transaction_id': [101, 102, 103],
    'transaction_sequence': [1, 2, 3]
}

df = pd.DataFrame(data)

# Convert timestamp to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract the date
df['date'] = df['timestamp'].dt.date

# Count transactions per day
transactions_per_day = df.groupby('date').size().reset_index(name='transaction_count')
print("Transactions per day:")
print(transactions_per_day)

# Define date range for filtering
start_date = pd.to_datetime('2023-09-01').date()
end_date = pd.to_datetime('2023-09-02').date()

# Filter and count transactions in the date range
filtered_df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
transactions_in_range = filtered_df.groupby('date').size().reset_index(name='transaction_count')

print("\nTransactions in the specified date range:")
print(transactions_in_range)
