import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("PJME_hourly.csv", parse_dates=["Datetime"], index_col="Datetime")

# Feature Engineering
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year

# Train/Test Split
train = df[:'2015']
test = df['2015':]
X_train, X_test = train.drop(columns=["PJME_MW"]), test.drop(columns=["PJME_MW"])
y_train, y_test = train["PJME_MW"], test["PJME_MW"]

# XGBoost Model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
xgb_model.fit(X_train, y_train)
preds_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
print(f'XGBoost RMSE: {rmse_xgb}')

# LSTM Model
X_train_lstm = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)
preds_lstm = lstm_model.predict(X_test_lstm).flatten()
rmse_lstm = np.sqrt(mean_squared_error(y_test, preds_lstm))
print(f'LSTM RMSE: {rmse_lstm}')

# Plot Results
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, preds_xgb, label='XGBoost Prediction')
plt.plot(y_test.index, preds_lstm, label='LSTM Prediction')
plt.legend()
plt.title('Energy Consumption Forecasting')
plt.show()
