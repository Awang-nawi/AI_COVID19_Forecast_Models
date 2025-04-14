#ARIMA#
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_excel("RESEARCH DATA.xlsx")
data.head()

# Filter the data to include only the relevant dates
filtered_data = data[(data['Date'] >= '2021-09-12') & (data['Date'] <= '2021-09-18')]

# Extract the 'New COVID-19 Cases' column
new_cases_vector_actual_data = filtered_data['New COVID-19 Cases'].values

print(new_cases_vector_actual_data)

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data to include only the relevant dates
data_filtered = data[(data['Date'] >= '2021-06-01') & (data['Date'] <= '2021-08-14')]

# Retain the Date column separately
dates = data_filtered['Date']

# Normalize the time series data excluding the Date column
scaler = MinMaxScaler()
time_series_data = data_filtered.drop(columns=['Date'])
normalized_data = pd.DataFrame(scaler.fit_transform(time_series_data), columns=time_series_data.columns)

# Add the Date column back to the normalized data
normalized_data['Date'] = dates.values

# Split the data into training and testing sets
train_set = normalized_data[(normalized_data['Date'] >= '2021-06-01') & (normalized_data['Date'] <= '2021-07-17')]
test_set = normalized_data[(normalized_data['Date'] >= '2021-07-18') & (normalized_data['Date'] <= '2021-08-14')]

# Extract the 'New COVID-19 Cases' column and convert it to a time series
covid_cases_ts = data_filtered.set_index('Date')['New COVID-19 Cases']

# Define the training period explicitly using the subset function
train_data = covid_cases_ts[:'2021-07-17']

# Fit ARIMA(9,2,2) model
arima_model = ARIMA(train_data, order=(9, 2, 2)).fit()

# Print the model summary
print(arima_model.summary())

# Forecast on the test set
test_data = covid_cases_ts['2021-07-18':'2021-08-14']
forecast = arima_model.forecast(steps=len(test_data))

# Calculate performance metrics
mse = mean_squared_error(test_data, forecast)
mae = mean_absolute_error(test_data, forecast)
mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
rmse = np.sqrt(mse)
rrmse = rmse / (np.max(test_data) - np.min(test_data))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Relative Root Mean Square Error (RRMSE): {rrmse}")

# Perform rolling forecasting between September 12, 2021 and September 18, 2021
start_date = '2021-09-12'
end_date = '2021-09-18'

rolling_forecast = []
history = list(train_data)

for date in pd.date_range(start=start_date, end=end_date):
    model = ARIMA(history, order=(9, 2, 2))
    model_fit = model.fit()
    output = model_fit.forecast()
    rolling_forecast.append(output[0])
    # For rolling forecast, append forecasted value to history
    history.append(output[0])

rolling_forecast_arima_with_no_exo = pd.Series(rolling_forecast, index=pd.date_range(start=start_date, end=end_date))

# Create the Series
rolling_forecast_arima_with_no_exo = pd.Series({
    '2021-09-12': 13178.841993,
    '2021-09-13': 13319.373523,
    '2021-09-14': 13450.319007,
    '2021-09-15': 14575.286429,
    '2021-09-16': 15269.166509,
    '2021-09-17': 16924.193064,
    '2021-09-18': 17110.933155
})

# Convert the Series to a NumPy vector
forecast_vector_arima_no_exo = rolling_forecast_arima_with_no_exo.values

print(forecast_vector_arima_no_exo)