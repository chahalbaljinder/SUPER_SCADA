import pandas as pd
import random
import string

# Generate timestamps
start_time = pd.to_datetime('2021-10-12 09:47:56')
end_time = pd.to_datetime('2021-10-13 19:00:05')
timestamps = pd.date_range(start=start_time, end=end_time, freq='5s')

# Generate other columns
EqN = [random.randint(100000, 999999) for _ in range(len(timestamps))]
Tseq = range(1, len(timestamps) + 1)
tid = [''.join(random.choices(string.ascii_uppercase + string.digits, k=6)) + str(i).zfill(6) for i in Tseq]

# Create DataFrame
df = pd.DataFrame({'timestamp': timestamps, 'EqN': EqN, 'Tseq': Tseq, 'tid': tid})

# Save to CSV
df.to_csv(r"C:\Users\admin\Desktop\metro ridership project\metro_transactions.csv", index=False)