# model_building.py

# 1. Imports & setup
# 2. Load split data (train/val/test)
# 3. ARIMA model (baseline)
# 4. SARIMA model
# 5. SARIMAX (with exogenous variables)
# 6. Evaluate + plot
# 7. Save predictions/metrics

# model_building.py

# --------------------------
# 1. Imports & Setup
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pmdarima

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from datetime import datetime


# --------------------------
# 2. Load the split datasets
# --------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')

# Load the saved merged dataset
merged_post_2008 = pd.read_csv(os.path.join(data_dir, "merged_dataset.csv"), parse_dates=["Date"])

# Manually define better split boundaries to expose the model to some exponential growth
val_start_date = datetime(2023, 1, 1)
test_start_date = datetime(2023, 9, 1)

# Split the dataset accordingly
train_df = merged_post_2008[merged_post_2008['Date'] < val_start_date].copy()
val_df = merged_post_2008[
    (merged_post_2008['Date'] >= val_start_date) &
    (merged_post_2008['Date'] < test_start_date)
].copy()
test_df = merged_post_2008[merged_post_2008['Date'] >= test_start_date].copy()

# Show summary of the new splits
(train_df.shape[0], train_df['Date'].min(), train_df['Date'].max()), \
(val_df.shape[0], val_df['Date'].min(), val_df['Date'].max()), \
(test_df.shape[0], test_df['Date'].min(), test_df['Date'].max())

"""
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'), parse_dates=['Date'])
val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'), parse_dates=['Date'])
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'), parse_dates=['Date'])

train_val_df = pd.concat([train_df, val_df])
"""
# --------------------------
# 3. ARIMA Model with auto_arima
# --------------------------
print("\nFitting ARIMA model using auto_arima...")

train_series = train_df['Price']
val_series = val_df['Price']

# Automatically determine best (p,d,q) parameters
stepwise_model = auto_arima(train_series,
                             start_p=1, start_q=1,
                             max_p=5, max_q=5,
                             seasonal=False,
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)

# Fit ARIMA model with best parameters
model = ARIMA(train_series, order=stepwise_model.order)
model_fit = model.fit()

# Forecast for the validation period
forecast = model_fit.forecast(steps=len(val_series))


model2 = ARIMA(train_series, order=(3, 0, 1))
model2_fit = model2.fit()
forecast2 = model2_fit.forecast(steps=len(val_series))

# --------------------------
# 4. Evaluate the model
# --------------------------
rmse = root_mean_squared_error(val_series, forecast)
mae = mean_absolute_error(val_series, forecast)

print(f"\nARIMA Model Evaluation on Validation Set:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")

rmse2 = root_mean_squared_error(val_series, forecast2)
mae2 = mean_absolute_error(val_series, forecast2)

print(f"\nARIMA Model Evaluation on Validation Set:")
print(f"RMSE: {rmse2:.2f}")
print(f"MAE:  {mae2:.2f}")

# --------------------------
# 5. Plot actual vs forecast
# --------------------------
plt.figure(figsize=(10,5))
plt.plot(val_df['Date'], val_series.values, label='Actual')
plt.plot(val_df['Date'], forecast.values, label='Forecast')
plt.title("ARIMA Forecast vs Actual (Validation)")
plt.xlabel("Date")
plt.ylabel("Cocoa Price")
plt.legend()
plt.tight_layout()
plt.show()  # << this one shows the first plot

plt.figure(figsize=(10,5))
plt.plot(val_df['Date'], val_series.values, label='Actual')
plt.plot(val_df['Date'], forecast2.values, label='Forecast2')
plt.title("ARIMA Forecast 2 vs Actual (Validation)")
plt.xlabel("Date")
plt.ylabel("Cocoa Price")
plt.legend()
plt.tight_layout()
plt.show()  # << this one shows the second






