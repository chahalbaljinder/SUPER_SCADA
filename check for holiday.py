import holidays
import pandas as pd

def get_indian_festivals(year, custom_festivals=None):
    # Get Indian holidays for the specified year
    indian_holidays = holidays.India(years=year)
    
    # Convert holidays to a DataFrame for easier manipulation
    holidays_df = pd.DataFrame(list(indian_holidays.items()), columns=['Date', 'Holiday'])
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])  # Ensure 'Date' is in datetime format

    # Print Indian holidays and festivals
    print(f"Indian Holidays and Festivals for {year}:")
    print(holidays_df)
    
    # Check if there are custom festivals to add
    if custom_festivals:
        # Convert custom festivals to DataFrame
        custom_festivals_df = pd.DataFrame(list(custom_festivals.items()), columns=['Date', 'Holiday'])
        custom_festivals_df['Date'] = pd.to_datetime(custom_festivals_df['Date'])
        
        # Append custom festivals to the existing holidays DataFrame
        combined_holidays_df = pd.concat([holidays_df, custom_festivals_df]).drop_duplicates()
        
        print(f"\nCombined Holidays and Custom Festivals for {year}:")
        print(combined_holidays_df)
    else:
        print(f"\nNo custom festivals to add for {year}.")

# Define the year you want to check
year = 2024

# Define additional custom festivals (optional)
custom_festivals = {
    '2024-01-14': 'Pongal',
    '2024-08-15': 'Independence Day',
    '2024-10-02': 'Gandhi Jayanti'
}

# Call the function to get Indian festivals and holidays
get_indian_festivals(year, custom_festivals)
