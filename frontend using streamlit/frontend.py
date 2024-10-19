import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data (weekday and weekend data)
weekday_data = pd.read_csv(r'C:\Users\admin\Desktop\airline\metro ridership project\metro_ridership_prediction\output of sensors\weekday_forecast_prophet_all_all.csv')
weekend_data = pd.read_csv(r'C:\Users\admin\Desktop\airline\metro ridership project\metro_ridership_prediction\output of sensors\weekend_forecast_prophet_all_all.csv')

# Function to plot graph
def plot_graph(data, start_date, end_date):
    print(data.columns)  # To verify column names
    data['timestamp'] = pd.to_datetime(data['timestamp'])  # Ensure correct date format

    # Filter the data based on the input dates
    filtered_data = data[(data['timestamp'] >= pd.to_datetime(start_date)) & (data['timestamp'] <= pd.to_datetime(end_date))]

    # Check if 'yhat' exists, otherwise use another column
    if 'transaction_id' in filtered_data.columns:
        y_column = 'transaction_id'
    elif 'y' in filtered_data.columns:
        y_column = 'y'  # Fall back to 'y' if 'yhat' doesn't exist
    else:
        raise KeyError("Neither 'yhat' nor 'y' found in the dataset.")

    # Plot the graph
    plt.plot(filtered_data['timestamp'], filtered_data[y_column], label='Predicted Footfall')
    plt.title(f'Footfall from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Footfall')
    plt.legend()
    plt.show()


# Streamlit UI
st.title('Footfall Prediction App')

# User input for start and end date
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

# Select the data type (weekday/weekend)
data_type = st.selectbox('Select Data Type', ['Weekday', 'Weekend'])

# Plot button
if st.button('Plot Graph'):
    if start_date and end_date:
        if data_type == 'Weekday':
            plot_graph(weekday_data, str(start_date), str(end_date))
        else:
            plot_graph(weekend_data, str(start_date), str(end_date))
    else:
        st.write('Please select valid start and end dates.')
