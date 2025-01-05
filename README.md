# Ridership Prediction

## Overview
The **SUPER SCADA (Supervisory Control and Data Acquisition)** system is an advanced platform designed for real-time monitoring and predictive analytics for the **Metro**. It aims to optimize operations, reduce delays, and ensure passenger safety by leveraging data from IoT devices, sensors, and control systems.

## Features
- **Real-Time Monitoring:**
  - Tracks train schedules and identifies potential delays.
  - Monitors critical equipment health and passenger safety conditions.
- **Predictive Analytics:**
  - Uses machine learning models to forecast operational metrics.
  - Identifies potential equipment failures and system inefficiencies.
- **Data Integration:**
  - Consolidates data from various departments into a unified platform.
- **Interactive Dashboards:**
  - Integration with Grafana for dynamic data visualization.

## Machine Learning Models
Three machine learning models were evaluated:
1. **LSTM:** Effective for sequential data but struggled with complex data relationships.
2. **XGBoost:** Efficient for structured data but limited in handling time series.
3. **N-BEATS:** Selected for its superior performance, achieving:
   - **Accuracy:** 81%
   - **F1 Score:** 72
   - **Error Rate:** 18%

## Challenges and Solutions
### Imbalanced Dataset:
- **Problem:** Rare events like equipment failures were underrepresented.
- **Solutions:**
  - Oversampling (SMOTE) and undersampling techniques.
  - Cost-sensitive learning.
  - Synthetic data generation for rare events.

### Data Quality:
- **Problem:** Missing values and noisy data.
- **Solutions:**
  - Implemented a robust preprocessing pipeline.
  - Engineered features like time-of-day and station occupancy.

## Performance Metrics
- **Accuracy:** 81%
- **Mean Absolute Error (MAE):** 105
- **R-Squared:** Demonstrated strong model fit.
- **Confusion Matrix:** Evaluated precision and recall for rare events.

## Future Enhancements
- **Feature Expansion:** Integrate external data (e.g., weather, special events).
- **Ensemble Models:** Combine models like Random Forest, XGBoost, and N-BEATS.
- **Real-Time Training:** Implement continuous learning with live data.
- **Predictive Maintenance:** Forecast equipment failures using health data.

## Scalability and Deployment
### Scalability:
- **Cloud-Based Infrastructure:** Use AWS or Azure for scaling.
- **Distributed Computing:** Apache Spark or Hadoop for parallel processing.
- **Edge Computing:** Local processing units to reduce latency.

### Deployment:
- **Version Control:** Robust CI/CD pipelines for seamless updates.
- **Real-Time Monitoring:** Tools like Prometheus and Grafana for performance alerts.
- **Security Measures:**
  - Data encryption.
  - Role-based access control (RBAC).

## Directory Structure
```
SUPER_SCADA/
├── data_exploration_filteration.ipynb
├── nbeats_model.pth
├── requirements.txt
├── python_script (incomplete).ipynb
├── rabbitmq_/
│   ├── config.py
│   ├── main_rabbitmq.py
│   └── trial_rabbitmq.py
├── Model_training.ipynb
└── newscaler.save
```

## Steps to Run
1. **Environment Setup:**
   - Install dependencies using `pip install -r requirements.txt`.

2. **Data Exploration:**
   - Update dataset path in `data_exploration_filteration.ipynb`.
   - Run the script to preprocess and clean data.

3. **Model Training:**
   - Update the dataset path in `Model_training.ipynb`.
   - Define hyperparameters:
     - `LOOKBACK = 1200`
     - `BATCH_SIZE = 64`
     - `EPOCHS = 100`
     - `LEARNING_RATE = 0.00001`
     - `DROPOUT_RATE = 0.2`
   - Train the model and save the outputs (model and scaler).

4. **Predictions:**
   - Run `main_rabbitmq.py` for predictions.
   - Use the Streamlit interface to input parameters (start date, end date, aggregation type) and view/download results.

5. **Visualization:**
   - Access interactive dashboards on Grafana to monitor real-time forecasts.

## Conclusion
The SUPER SCADA system demonstrates how predictive modeling can transform metro operations by:
- Enhancing operational efficiency.
- Improving passenger safety.
- Providing actionable insights through advanced analytics.

Future iterations will focus on scalability, integrating additional data sources, and refining model performance to meet evolving urban transportation challenges.

