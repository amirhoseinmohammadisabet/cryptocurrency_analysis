import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Load the CSV file into a DataFrame
data = pd.read_csv('Data/cryptocurrency_prices_cluster.csv', parse_dates=['Date'], index_col='Date')

# Assuming the first column contains the names of cryptocurrencies
# and the remaining columns contain the prices over time
cryptos = data.columns

# Select the cryptocurrencies you're interested in
selected_cryptos = ['MKR-USD', 'BTC-USD', 'LINK-USD', 'YFI-USD']

# Generate buy and sell signals for each selected cryptocurrency
signals = {}

for crypto in selected_cryptos:
    try:
        print(f"Generating buy and sell signals for {crypto}...")
        
        # Get the price data for the current cryptocurrency
        crypto_prices = data[crypto]
        
        # Convert prices to numeric type
        crypto_prices = pd.to_numeric(crypto_prices, errors='coerce')
        
        # Remove any missing values
        crypto_prices = crypto_prices.dropna()
        
        # Forecasting for the next week, two weeks, and a month
        forecast_days = [7, 14, 30]
        forecasts = {}
        
        for days in forecast_days:
            end_date = crypto_prices.index[-1] + timedelta(days=days)
            forecast_dates = pd.date_range(start=crypto_prices.index[-1], end=end_date)
            
            # Train ARIMA model
            model = ARIMA(crypto_prices, order=(5,1,0))
            model_fit = model.fit()
            
            # Forecast prices
            forecast = model_fit.forecast(steps=days)
            forecasts[days] = forecast
        
        # Calculate moving averages
        short_window = 10
        long_window = 50
        
        crypto_prices['Short_MA'] = crypto_prices.rolling(window=short_window, min_periods=1).mean()
        crypto_prices['Long_MA'] = crypto_prices.rolling(window=long_window, min_periods=1).mean()
        
        # Generate buy and sell signals based on forecasts and moving averages
        buy_sell_signals = {}
        
        for days, forecast in forecasts.items():
            next_price = forecast[0]
            current_price = crypto_prices[-1]
            
            short_ma = crypto_prices['Short_MA'].iloc[-1]
            long_ma = crypto_prices['Long_MA'].iloc[-1]
            
            if next_price > current_price and short_ma > long_ma:
                signal = 'BUY'
            elif next_price < current_price and short_ma < long_ma:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            buy_sell_signals[days] = signal
        
        signals[crypto] = buy_sell_signals
    except Exception as e:
        print(f"Error processing {crypto}: {e}")

# Print the generated signals
print("\nBuy and Sell Signals:")
for crypto, signal_data in signals.items():
    print(f"\nCrypto: {crypto}")
    for days, signal in signal_data.items():
        print(f"{days}-day Forecast: {signal}")
