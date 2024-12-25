#pip install pika

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import pika
from datetime import timedelta
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from config import RABBITMQ_HOST, RABBITMQ_PORT ,QUEUE_NAME, EXCHANGE, ROUTING_KEY , USERNAME , PASSWORD 

# Function to publish a dummy message to RabbitMQ
def publish_to_rabbitmq(json_message):
    # Set up credentials for the connection
    credentials = pika.PlainCredentials(USERNAME, PASSWORD)

    # Establish a connection to RabbitMQ with the given credentials
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials
        )
    )
    channel = connection.channel()

    # Declare the queue to ensure it exists
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    
    # Convert the message to JSON format
    #json_message = json.dumps(message)

    # Publish the message to the queue
    channel.basic_publish(
        exchange=EXCHANGE,
        routing_key=ROUTING_KEY,
        body=json_message,
        properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
    )

    print(f"Message published to queue '{QUEUE_NAME}': {json_message}")

    # Close the connection
    connection.close()

# Rest of your code (model class, forecast function, etc.) goes here
# Load N-BEATS Model Class
class NBeatsModel(nn.Module):
    def __init__(self, input_size, hidden_dim=256, dropout_rate=0.2):
        super(NBeatsModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        forecast = self.fc4(x)
        return forecast

# Function to forecast future transactions
def forecast_future_transactions(model_path, scaler_path, hourly_transaction, start_date, end_date, lookback=1200):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    num_hours = int((end_date - start_date).total_seconds() // 3600)

    # Load the scaler and model
    scaler = joblib.load(scaler_path)
    loaded_model = NBeatsModel(input_size=lookback)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    # Prepare input sequence from most recent data
    recent_data = hourly_transaction[-lookback:].values
    scaled_recent_data = scaler.transform(recent_data.reshape(-1, 1))
    X_input = torch.tensor(scaled_recent_data.reshape(1, -1), dtype=torch.float32)

    # Generate predictions
    future_dates = pd.date_range(start=start_date, periods=num_hours, freq='H')
    future_predictions = []
    
    for _ in range(num_hours):
        with torch.no_grad():
            forecast = loaded_model(X_input).item()
        future_predictions.append(forecast)
        # Update the input sequence
        X_input = torch.cat([X_input[:, 1:], torch.tensor([[forecast]])], dim=1)

    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Dt': future_dates,
        'Predicted_Transaction_Count': future_predictions_rescaled.flatten()
    })
    
    return forecast_df

# Aggregation function
def aggregate_forecast(forecast_df, aggregation_type):
    # Set the date column as index
    forecast_df.set_index('Dt', inplace=True)
    
    if aggregation_type == 'Hourly':
        return forecast_df['Predicted_Transaction_Count'].resample('H').sum()
    elif aggregation_type == 'Daily':
        return forecast_df['Predicted_Transaction_Count'].resample('D').sum()
    elif aggregation_type == 'Day of Month':
        return forecast_df.groupby(forecast_df.index.day)['Predicted_Transaction_Count'].sum()
    elif aggregation_type == 'Week of Year':
        return forecast_df.groupby(forecast_df.index.isocalendar().week)['Predicted_Transaction_Count'].sum()
    elif aggregation_type == 'Week of Month':
        return forecast_df.groupby((forecast_df.index.to_period('M').astype(str) + "-" + forecast_df.index.to_period('W').astype(str))).sum()
    elif aggregation_type == 'Monthly':
        return forecast_df['Predicted_Transaction_Count'].resample('M').sum()


# Streamlit interface
def main():
    st.title("N-BEATS Transaction Count Forecast")

    # Input start and end date for forecasting
    start_date = st.text_input("Enter Start Date (YYYY-MM-DD HH:MM:SS)", "2024-10-05 00:00:00")
    end_date = st.text_input("Enter End Date (YYYY-MM-DD HH:MM:SS)", "2024-10-30 23:59:00")
    
    # Dropdown to select aggregation type
    aggregation_type = st.selectbox(
        "Select Aggregation Type",
        ["Hourly", "Daily", "Day of Month", "Week of Year", "Week of Month", "Monthly"]
    )

    if st.button('Generate Forecast'):
        # Load your dataset for hourly transactions (replace this with your actual data)
        df = pd.read_csv(r'Aggregated_Data.csv')
        df['Dt'] = pd.to_datetime(df['Dt'], errors='coerce')
        df = df.dropna(subset=['Dt'])
        df.set_index('Dt', inplace=True)
        hourly_transaction = df.resample('H').size()

        # Forecast future transactions
        model_path = r'nbeats_model.pth'
        scaler_path = r'newscaler.save'

        try:
            forecast_df = forecast_future_transactions(model_path, scaler_path, hourly_transaction, start_date, end_date)
            
            # Display forecast data
            st.write("Forecasted Data:")
            st.write(forecast_df)

            # Aggregation based on dropdown selection
            aggregated_data = aggregate_forecast(forecast_df, aggregation_type)
            
            # Display aggregated results
            st.write(f"{aggregation_type} Aggregation:")
            st.write(aggregated_data)

            # Plot forecast vs actual
            st.write(f"Forecast Plot ({aggregation_type}):")
            plt.figure(figsize=(10, 6))
            plt.plot(aggregated_data.index, aggregated_data.values, label=f'N-BEATS {aggregation_type} Forecast')
            plt.title(f"Forecasted Transaction Counts ({aggregation_type})")
            plt.legend()
            st.pyplot(plt)

            # # Prepare the message for RabbitMQ
            # message = {
            #     "forecast": forecast_df.to_dict(orient='records'),
            #     "aggregation_type": aggregation_type,
            #     "aggregated_data": aggregated_data.to_dict(),
            #     "start_date": start_date,
            #     "end_date": end_date
            # }

            # Ensure 'Dt' is properly set when creating forecast_df
            forecast_df.reset_index(inplace=True)  # Ensure 'Dt' is a column, not the index


            # Prepare the message for RabbitMQ
            message = {
                "forecast": forecast_df.assign(Dt=forecast_df['Dt'].dt.strftime('%Y-%m-%d %H:%M:%S')).to_dict(orient='records'),
                "aggregation_type": aggregation_type,
                "aggregated_data": {str(k): v for k, v in aggregated_data.to_dict().items()},
                "start_date": start_date,
                "end_date": end_date
            }

            # Convert the message to JSON format
            json_message = json.dumps(message)

            # Publish to RabbitMQ
            publish_to_rabbitmq(json_message)
            st.success(f"Data successfully published to RabbitMQ queue: {QUEUE_NAME}")

        except ValueError as e:
            st.error(f"Error during forecasting: {e}")

if __name__ == "__main__":
    main()

