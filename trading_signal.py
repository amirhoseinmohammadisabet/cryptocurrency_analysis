import yfinance as yf
from prophet import Prophet
import pandas as pd

# Step 1: Fetch historical Bitcoin price data using Yahoo Finance
btc_data = yf.download('BTC-USD', start='2023-01-01', end='2023-12-31')

# Step 2: Preprocess the data for Prophet
btc_data.reset_index(inplace=True)
btc_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
btc_data = btc_data[['ds', 'y']]

# Step 3: Train a Prophet model
prophet_model = Prophet()
prophet_model.fit(btc_data)

# Step 4: Forecast future prices
future = prophet_model.make_future_dataframe(periods=30)  # Forecasting for the next 30 days
forecast = prophet_model.predict(future)

# Step 5: Generate buy and sell signals with risk management
buy_threshold = 0.03  # Buy when the forecasted price increases by at least 3%
sell_threshold = -0.03  # Sell when the forecasted price decreases by at least 3%
stop_loss = -0.05  # Apply a stop-loss at a 5% loss

# Initialize balance and buy price
initial_balance = 1000
balance = initial_balance
buy_price = 0

# Iterate through forecasted prices and generate signals
for index, row in forecast.iterrows():
    if row['yhat_upper'] is not None and row['yhat_lower'] is not None:
        upper_price = row['yhat_upper']
        lower_price = row['yhat_lower']
        if row['yhat'] - lower_price > buy_threshold and buy_price == 0:
            buy_price = row['yhat']
            print("Buy at:", row['ds'], "Price:", buy_price)
        elif row['yhat'] - upper_price < sell_threshold and buy_price > 0:
            profit = row['yhat'] - buy_price
            balance += profit
            print("Sell at:", row['ds'], "Price:", row['yhat'], "Profit:", profit)
            buy_price = 0
        elif row['yhat'] - lower_price < stop_loss and buy_price > 0:
            loss = buy_price - row['yhat']
            balance -= loss
            print("Stop-loss triggered at:", row['ds'], "Price:", row['yhat'], "Loss:", loss)
            buy_price = 0

# Display final balance
print("Initial Balance:", initial_balance)
print("Final Balance:", balance)
print("Profit/Loss:", balance - initial_balance)
