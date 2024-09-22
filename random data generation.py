import pandas as pd
import numpy as np

# Load the dataset
file_path = r'C:\Users\admin\Desktop\airline\sensor-file-ridership/transaction.csv'
data = pd.read_csv(file_path)

# Generate the new columns

# 1. Reference Seq_no: Sequence from 1 to the length of the data
data['Reference Seq_no'] = np.arange(1, len(data) + 1)

# 2. DSM: Randomly assign one of 20-25 unique 5-digit integers across the rows
unique_dsm_values = np.random.choice(range(10000, 99999), size=np.random.randint(20, 26), replace=False)
data['DSM'] = np.random.choice(unique_dsm_values, size=len(data), replace=True)

# 3. Tag: Randomly assign one of 30-35 tags from a given list (e.g., 2001, 2002, 2003, 2004, 5001)
tag_values = [2001, 2002, 2003, 2004, 5001]
tag_pool = np.random.choice(tag_values, size=np.random.randint(30, 36), replace=True)
data['Tag'] = np.random.choice(tag_pool, size=len(data), replace=True)

# Save the updated data to a new file
output_path = r'C:\Users\admin\Desktop\airline\sensor-file-ridership/transaction_updated.csv'
data.to_csv(output_path, index=False)

output_path
