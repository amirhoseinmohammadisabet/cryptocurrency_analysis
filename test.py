import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('Data/data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
data.index.freq = 'D' 

target_variable = 'price_btc'

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Train ARIMA model
order = (10, 20, 30)  
model = ARIMA(train[target_variable], order=order)
fit_model = model.fit()

# Make predictions for the next month
forecast_steps = 30 
forecast = fit_model.get_forecast(steps=forecast_steps)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(train.index, train[target_variable], label='Training Data')
plt.plot(test.index, test[target_variable], label='Testing Data')
plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()
