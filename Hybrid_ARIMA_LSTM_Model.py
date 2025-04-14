import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt



# Load the dataset
data = pd.read_excel("RESEARCH DATA.xlsx")

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data to include only the relevant dates
data_filtered = data[(data['Date'] >= '2021-06-01') & (data['Date'] <= '2021-08-14')]

# Extract the 'New COVID-19 Cases' column
covid_cases = data_filtered['New COVID-19 Cases'].values

# Train ARIMA model
arima_model = ARIMA(covid_cases, order=(9, 2, 2))
arima_fit = arima_model.fit()

# Get residuals from ARIMA model
residuals = arima_fit.resid

# Normalize the residuals
scaler = MinMaxScaler()
residuals_scaled = scaler.fit_transform(residuals.reshape(-1, 1))

# Define sequence length
seq_length = 10

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

# Create sequences for training
x_train, y_train = create_sequences(residuals_scaled, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=200, input_shape=(seq_length, 1)))
model.add(Dense(units=1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam())

# Train the model
history = model.fit(x_train, y_train, epochs=472, batch_size=22, verbose=1)

# Print the model summary
print(model.summary())

# Load the dataset
data = pd.read_excel("RESEARCH DATA.xlsx")

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data to include only the relevant dates
data_filtered = data[(data['Date'] >= '2021-06-01') & (data['Date'] <= '2021-08-14')]

# Extract the 'New COVID-19 Cases' column
covid_cases = data_filtered['New COVID-19 Cases'].values

# Split the data into training and test sets
train_end_date = '2021-07-17'
test_start_date = '2021-07-18'

train_data = data_filtered[data_filtered['Date'] <= train_end_date]['New COVID-19 Cases'].values
test_data = data_filtered[data_filtered['Date'] >= test_start_date]['New COVID-19 Cases'].values

# Train ARIMA model
arima_model = ARIMA(train_data, order=(9, 2, 2))
arima_fit = arima_model.fit()

# Get residuals from ARIMA model
train_residuals = arima_fit.resid

# Normalize the residuals
scaler = MinMaxScaler()
train_residuals_scaled = scaler.fit_transform(train_residuals.reshape(-1, 1))

# Define sequence length
seq_length = 10

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

# Create sequences for training
x_train, y_train = create_sequences(train_residuals_scaled, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=200, input_shape=(seq_length, 1)))
model.add(Dense(units=1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam())

# Train the model
history = model.fit(x_train, y_train, epochs=472, batch_size=22, verbose=1)

# Predict the test data with ARIMA
arima_forecast = arima_fit.forecast(steps=len(test_data))
arima_residuals = test_data - arima_forecast

# Normalize the ARIMA residuals
arima_residuals_scaled = scaler.transform(arima_residuals.reshape(-1, 1))

# Create sequences for LSTM predictions
x_test, y_test = create_sequences(arima_residuals_scaled, seq_length)

# Predict the residuals with LSTM
lstm_forecast_scaled = model.predict(x_test)

# Rescale the LSTM predictions
lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled)

# Combine ARIMA forecast and LSTM residual forecast
final_forecast = arima_forecast[seq_length:] + lstm_forecast.flatten()

# Calculate performance metrics
mse = mean_squared_error(test_data[seq_length:], final_forecast)
mae = mean_absolute_error(test_data[seq_length:], final_forecast)
mape = np.mean(np.abs((test_data[seq_length:] - final_forecast) / test_data[seq_length:])) * 100
rmse = np.sqrt(mse)
rrmse = rmse / (np.max(test_data[seq_length:]) - np.min(test_data[seq_length:]))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Relative Root Mean Square Error (RRMSE): {rrmse}")

# Perform rolling forecasting between September 12, 2021, and September 18, 2021
forecast_start_date = '2021-09-12'
forecast_end_date = '2021-09-18'
forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date)

# Initialize history with the last `seq_length` sequences from training data
history_cases = list(train_data[-seq_length:])
history_residuals = list(train_residuals_scaled[-seq_length:])
history_residuals = [seq[0] for seq in history_residuals]  # Ensure it's a list of scalar values

rolling_forecast = []

for date in forecast_dates:
    # Use the ARIMA model to forecast the next value
    arima_forecast_next = arima_fit.forecast(steps=1)[0]
    history_cases.append(arima_forecast_next)
    history_cases = history_cases[-seq_length:]  # Keep the length of history consistent

    # Calculate the residuals
    arima_residual_next = arima_forecast_next - (history_cases[-2] if len(history_cases) > 1 else arima_forecast_next)
    history_residuals.append(scaler.transform([[arima_residual_next]])[0][0])
    history_residuals = history_residuals[-seq_length:]  # Keep the length of history consistent

    # Predict the residuals with LSTM
    input_seq = np.array(history_residuals).reshape((1, seq_length, 1))
    lstm_forecast_next_scaled = model.predict(input_seq)
    lstm_forecast_next = scaler.inverse_transform(lstm_forecast_next_scaled)[0, 0]

    # Combine ARIMA forecast and LSTM residual forecast
    final_forecast_next = arima_forecast_next + lstm_forecast_next
    rolling_forecast.append(final_forecast_next)

# Rescale the forecasted values back to the original scale
rolling_forecast_rescaled = scaler.inverse_transform(np.array(rolling_forecast).reshape(-1, 1))

# Create a DataFrame for plotting
rolling_forecast_combined_no_exo_df = pd.DataFrame(data=rolling_forecast_rescaled, index=forecast_dates, columns=['Forecast'])
rolling_forecast_combined_no_exo_df['Forecast'] = np.abs(rolling_forecast_combined_no_exo_df['Forecast'])

import numpy as np
import pandas as pd

# Create the DataFrame
rolling_forecast_combined_no_exg_df = pd.DataFrame({
    'Forecast': [
        13153,
        13189,
        13213,
        13216,
        13199,
        13400,
        14263
    ]
}, index=pd.date_range(start='2021-09-12', end='2021-09-18'))

# Convert the DataFrame to a NumPy vector
forecast_vector_combined_no_exg = rolling_forecast_combined_no_exg_df['Forecast'].values

print(forecast_vector_combined_no_exg)