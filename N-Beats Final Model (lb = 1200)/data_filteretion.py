import polars as pl
import os
import pandas as pd  # Ensure you have pandas imported for date handling
from datetime import datetime, timedelta

def process_time_series_data_polars(file_path):
    # Get today's date and cutoff date (one day before today)
    today = datetime.now().date()
    cutoff_date = today - timedelta(days=1)

    # Load the data using Polars and convert 'Dt' to datetime with a specific format
    #df = pl.read_csv(file_path, dtypes={"Dt": pl.Date})

    # If you want to ensure correct date format, you can first load it with pandas:
    df_pandas = pd.read_csv(file_path)
    df_pandas['Dt'] = pd.to_datetime(df_pandas['Dt'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')  # Adjust format as needed
    df = pl.from_pandas(df_pandas)  # Convert back to Polars DataFrame if needed

    # Filter the data where 'Dt' is less than the cutoff date
    df_filtered = df.filter(pl.col("Dt") < pl.lit(cutoff_date))

    # Sort the data by 'Dt'
    df_sorted = df_filtered.sort("Dt")

    # Group by station and count entries
    station_counts = df_sorted.groupby("Sta").agg(pl.count("Sta").alias("Count"))

    print("Station Entry Counts:")
    print(station_counts)

    # Ask the user if they want to create CSVs for each station
    save_csvs = input("Do you want to save CSV files for each station? (yes/no): ").strip().lower()

    if save_csvs == 'yes':
        # Create a directory for the station CSVs
        output_dir = "station_csvs_polars"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save a separate CSV for each station
        for sta in station_counts['Sta']:
            station_data = df_sorted.filter(pl.col("Sta") == sta)
            count = station_counts.filter(pl.col("Sta") == sta)['Count'][0]
            file_name = f"{sta}_entries_{count}.csv"
            file_path = os.path.join(output_dir, file_name)
            station_data.write_csv(file_path)
            print(f"Saved CSV for station {sta} with {count} entries at {file_path}")

    # Find missing dates in the aggregated data
    min_date = df_sorted['Dt'].min().date()
    max_date = df_sorted['Dt'].max().date()

    # Generate a full date range between the min and max dates
    full_date_range = pl.date_range(min_date, max_date, "1d")

    # Get the set of unique dates present in the dataset
    present_dates = df_sorted['Dt'].unique()

    # Find the missing dates by comparing the full date range with present dates
    missing_dates = full_date_range.filter(~full_date_range.is_in(present_dates))

    print("\nMissing Dates:")
    print(missing_dates)

    # Calculate the count of missing days
    missing_days_count = missing_dates.shape[0]
    print(f"\nCount of Missing Days: {missing_days_count}")

    # Find missing date ranges (consecutive missing days)
    missing_date_ranges = []
    if missing_days_count > 0:
        start_date = missing_dates[0]
        end_date = missing_dates[0]

        for i in range(1, missing_days_count):
            if missing_dates[i] == missing_dates[i - 1] + timedelta(days=1):
                end_date = missing_dates[i]
            else:
                missing_date_ranges.append((start_date, end_date))
                start_date = missing_dates[i]
                end_date = missing_dates[i]
        # Append the last range
        missing_date_ranges.append((start_date, end_date))

    print("\nMissing Date Ranges:")
    for start, end in missing_date_ranges:
        print(f"From {start} to {end} ({(end - start).days + 1} days)")

    return df_sorted, station_counts, missing_dates, missing_date_ranges

# Use the file path to your dataset
file_path = r'C:\Users\admin\Desktop\super_scada\N-Beats Final Model (lb = 1200)\Aggregated_Data.csv'  # Replace with the actual file path
processed_df_polars, station_entry_counts_polars, missing_dates_polars, missing_ranges_polars = process_time_series_data_polars(file_path)
