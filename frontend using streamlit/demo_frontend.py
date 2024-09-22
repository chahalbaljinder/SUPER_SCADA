import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files based on the data type selected
file_paths = {
    'daily': r'C:\Users\admin\Desktop\airline\sensor-file-ridership\output of sensors\daily_forecast_prophet.csv',
    'weekly': r'C:\Users\admin\Desktop\airline\sensor-file-ridership\output of sensors\weekly_forecast_prophet.csv',
    'monthly': r'C:\Users\admin\Desktop\airline\sensor-file-ridership\output of sensors\monthly_forecast_prophet.csv',
    'yearly': r'C:\Users\admin\Desktop\airline\sensor-file-ridership\output of sensors\yearly_forecast_prophet.csv',
    'weekend': r'C:\Users\admin\Desktop\airline\sensor-file-ridership\output of sensors\weekend_forecast_prophet.csv',
    'weekday': r'C:\Users\admin\Desktop\airline\sensor-file-ridership\output of sensors\weekday_forecast_prophet.csv'
}

# Function to load and display data
def load_data(data_type):
    df = pd.read_csv(file_paths[data_type])
    return df

# Function to plot data
def plot_data(df, data_type):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df)
    plt.title(f"{data_type.capitalize()} Data")
    plt.xlabel("Date")
    plt.ylabel("Value")
    st.pyplot(plt)

# Streamlit UI
st.title("Time Series Data Visualization")

# Sidebar options for data types
data_type = st.sidebar.selectbox(
    "Select Data Type",
    ('daily', 'weekly', 'monthly', 'yearly', 'weekend', 'weekday', 'weekday_vs_weekend')
)

# Load the selected data
df = load_data(data_type)

# Display the plot
st.write(f"### {data_type.capitalize()} Data")
plot_data(df, data_type)

# Provide download options for CSV and image
csv = df.to_csv(index=False)
st.download_button(label="Download CSV", data=csv, file_name=f"{data_type}_data.csv", mime='text/csv')

# Save plot as an image and provide download option
plt.savefig('plot.png')
st.download_button(label="Download Image", data=open('plot.png', 'rb'), file_name="plot.png", mime="image/png")
