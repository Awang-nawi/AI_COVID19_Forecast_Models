import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Concatenate, Input, Flatten
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf

# Load the dataset
data = pd.read_excel("RESEARCH DATA.xlsx")

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data to include only the relevant dates
data_filtered = data[(data['Date'] >= '2021-06-01') & (data['Date'] <= '2021-08-14')]

# Extract the 'New COVID-19 Cases' column and normalize the data
covid_cases = data_filtered['New COVID-19 Cases'].values
scaler = MinMaxScaler()
covid_cases_scaled = scaler.fit_transform(covid_cases.reshape(-1, 1))

# Define the training period explicitly
train_end_date = '2021-07-17'
test_start_date = '2021-07-18'
test_end_date = '2021-08-14'

train_data = data_filtered[data_filtered['Date'] <= train_end_date]['New COVID-19 Cases'].values
test_data = data_filtered[(data_filtered['Date'] >= test_start_date) & (data_filtered['Date'] <= test_end_date)]['New COVID-19 Cases'].values

# Normalize train and test data
train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
test_data_scaled = scaler.transform(test_data.reshape(-1, 1))

print(f"train_data length: {len(train_data_scaled)}, test_data length: {len(test_data_scaled)}")

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

# Define sequence length
seq_length = 10

# Create sequences for training and testing
x_train, y_train = create_sequences(train_data_scaled, seq_length)
x_test, y_test = create_sequences(test_data_scaled, seq_length)

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=200, input_shape=(seq_length, 1)))
model.add(Dense(units=1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=472, batch_size=22, verbose=1)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
data = pd.read_excel("RESEARCH DATA.xlsx")

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data to include only the relevant dates
data_filtered = data[(data['Date'] >= '2021-06-01') & (data['Date'] <= '2021-08-14')]

# Extract the 'New COVID-19 Cases' column and normalize the data
covid_cases = data_filtered['New COVID-19 Cases'].values
scaler = MinMaxScaler()
covid_cases_scaled = scaler.fit_transform(covid_cases.reshape(-1, 1))

# Define the training period explicitly
train_end_date = '2021-07-17'
test_start_date = '2021-07-18'
test_end_date = '2021-08-14'

train_data = data_filtered[data_filtered['Date'] <= train_end_date]['New COVID-19 Cases'].values
test_data = data_filtered[(data_filtered['Date'] >= test_start_date) & (data_filtered['Date'] <= test_end_date)]['New COVID-19 Cases'].values

# Normalize train and test data
train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
test_data_scaled = scaler.transform(test_data.reshape(-1, 1))

print(f"train_data length: {len(train_data_scaled)}, test_data length: {len(test_data_scaled)}")

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

# Define sequence length
seq_length = 10

# Create sequences for training and testing
x_train, y_train = create_sequences(train_data_scaled, seq_length)
x_test, y_test = create_sequences(test_data_scaled, seq_length)

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=200, input_shape=(seq_length, 1)))
model.add(Dense(units=1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=22, verbose=1)  # Reduced epochs for quicker debugging

# Forecasting
y_pred = model.predict(x_test)
print(f"y_pred shape: {y_pred.shape}")

# Rescale the predicted and true values back to the original scale
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_rescaled = scaler.inverse_transform(y_pred)

# Calculate performance metrics
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
rmse = np.sqrt(mse)
rrmse = rmse / (np.max(y_test_rescaled) - np.min(y_test_rescaled))

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
history = list(train_data_scaled[-seq_length:])
history = [seq[0] for seq in history]  # Ensure it's a list of scalar values
rolling_forecast = []

for date in forecast_dates:
    input_seq = np.array(history[-seq_length:]).reshape((1, seq_length, 1))
    forecast = model.predict(input_seq)
    rolling_forecast.append(forecast[0, 0])
    history.append(forecast[0, 0])

# Rescale the forecasted values back to the original scale
rolling_forecast_rescaled = scaler.inverse_transform(np.array(rolling_forecast).reshape(-1, 1))

# Create a DataFrame for plotting
rolling_forecast_lstm_no_exo_df = pd.DataFrame(data=rolling_forecast_rescaled, index=forecast_dates, columns=['Forecast'])
rolling_forecast_lstm_no_exo_df['Forecast']=np.abs(rolling_forecast_lstm_no_exo_df['Forecast'])

rolling_forecast_lstm_no_exo_df
import numpy as np
import pandas as pd

# Create the DataFrame
rolling_forecast_lstm_no_exo_df = pd.DataFrame({
    'Forecast': [
        12554.629883,
        10150.577148,
        8157.168457,
        5648.877930,
        2569.294922,
        1413.408691,
        4763.826660
    ]
}, index=pd.date_range(start='2021-09-12', end='2021-09-18'))

# Convert the DataFrame to a NumPy vector
forecast_vector_lstm_no_exo = rolling_forecast_lstm_no_exo_df['Forecast'].values

print(forecast_vector_lstm_no_exo)