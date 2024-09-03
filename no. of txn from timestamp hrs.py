import pandas as pd

# Sample DataFrame
data = {
    'timestamp': ['2024-09-01 10:00:00', '2024-09-01 11:30:00', '2024-09-01 12:15:00', 
                  '2024-09-01 14:00:00', '2024-09-02 09:00:00'],
    'transaction_id': [1, 2, 3, 4, 5],
    'transaction_sequence': [101, 102, 103, 104, 105]
}

df = pd.DataFrame(data)

# Convert 'timestamp' to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Define the time period for which you want to count transactions
start_time = '2024-09-01 10:00:00'
end_time = '2024-09-01 13:00:00'

# Filter transactions within the specified time range
filtered_transactions = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

# Count the number of transactions
num_transactions = filtered_transactions.shape[0]  # or len(filtered_transactions)

print(f'Number of transactions made between {start_time} and {end_time}: {num_transactions}')
