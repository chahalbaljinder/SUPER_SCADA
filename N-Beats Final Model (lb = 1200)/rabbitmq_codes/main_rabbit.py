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
from config import RABBITMQ_URL, QUEUE_NAME, EXCHANGE, ROUTING_KEY

# RabbitMQ Publish Function
def publish_to_rabbitmq(message):
    # Ensure message is JSON formatted
    json_message = json.dumps(message)
    
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_URL))
    channel = connection.channel()
    
    # Declare queue in case it doesn't exist
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    
    # Publish JSON message to the queue
    channel.basic_publish(
        exchange=EXCHANGE,
        routing_key=ROUTING_KEY,
        body=json_message,
        properties=pika.BasicProperties(delivery_mode=2)  # make message persistent
    )
    
    connection.close()

# Rest of your code (model class, forecast function, etc.) goes here

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

            # Prepare the message for RabbitMQ
            message = {
                "forecast": forecast_df.to_dict(orient='records'),
                "aggregation_type": aggregation_type,
                "aggregated_data": aggregated_data.to_dict(),
                "start_date": start_date,
                "end_date": end_date
            }

            # Publish to RabbitMQ
            publish_to_rabbitmq(message)
            st.success(f"Data successfully published to RabbitMQ queue: {QUEUE_NAME}")

        except ValueError as e:
            st.error(f"Error during forecasting: {e}")

if __name__ == "__main__":
    main()

