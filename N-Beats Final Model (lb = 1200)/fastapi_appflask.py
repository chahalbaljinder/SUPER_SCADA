import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from datetime import datetime
import torch.nn as nn
import matplotlib.pyplot as plt

# Hyperparameters
LOOKBACK = 1200

# Updated N-BEATS Model Definition with Dropout
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
def forecast_future_transactions(model_path, scaler_path, hourly_transaction, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    num_hours = int((end_date - start_date).total_seconds() // 3600)
    if num_hours <= 0:
        raise ValueError("End date must be after start date.")

    # Load the scaler
    scaler = joblib.load(scaler_path)
    
    recent_data = hourly_transaction[-LOOKBACK:].values
    if len(recent_data) < LOOKBACK:
        raise ValueError("Not enough recent data to create the input sequence.")

    scaled_recent_data = scaler.transform(recent_data.reshape(-1, 1))
    X_input = torch.tensor(scaled_recent_data[-LOOKBACK:].reshape(1, -1), dtype=torch.float32)

    future_dates = pd.date_range(start=start_date, periods=num_hours, freq='H')
    future_predictions = []

    # Load the model
    loaded_model = NBeatsModel(input_size=LOOKBACK)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    for _ in range(num_hours):
        with torch.no_grad():
            forecast = loaded_model(X_input).item()

        future_predictions.append(forecast)
        X_input = torch.cat([X_input[:, 1:], torch.tensor([[forecast]])], dim=1)

    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    forecast_df = pd.DataFrame({
        'Dt': future_dates,
        'Predicted_Transaction_Count': future_predictions_rescaled.flatten()
    })
    
    return forecast_df

# Streamlit app
def main():
    st.title('Transaction Forecasting App')
    
    # Input start and end dates
    start_date = st.text_input("Enter start date (e.g., 2024-10-05 00:00:00):")
    end_date = st.text_input("Enter end date (e.g., 2024-10-30 23:59:00):")
    
    if st.button("Run Forecast"):
        if not start_date or not end_date:
            st.error("Please enter both start and end dates.")
        else:
            try:
                # Path to model and scaler
                model_path = 'nbeats_model.pth'
                scaler_path = 'newscaler.save'

                # Load sample hourly transaction data
                df = pd.read_csv('Aggregated_Data.csv')  # Use your dataset
                df['Dt'] = pd.to_datetime(df['Dt'], errors='coerce')
                df = df.dropna(subset=['Dt'])
                df.set_index('Dt', inplace=True)
                hourly_transaction = df.resample('H').size()

                # Perform forecasting
                forecast_df = forecast_future_transactions(model_path, scaler_path, hourly_transaction, start_date, end_date)
                
                # Display forecasted data
                st.write(forecast_df.head())
                
                # Plotting
                st.line_chart(forecast_df.set_index('Dt')['Predicted_Transaction_Count'])

                # Save predictions to CSV
                forecast_df.to_csv('forecasted_transactions.csv', index=False)
                st.success("Forecast saved to 'forecasted_transactions.csv'")

            except Exception as e:
                st.error(f"Error: {e}")

# Run the app
if __name__ == "__main__":
    main()
