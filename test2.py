import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# Load the CSV file into a DataFrame
data = pd.read_csv('Data/cryptocurrency_prices_cluster.csv')

# Assuming the first column contains the names of cryptocurrencies
# and the remaining columns contain the prices over time
cryptos = data.iloc[:, 0]
prices = data.iloc[:, 1:]

# Select the 4 cryptocurrencies you're interested in
selected_cryptos = ['ETH-USD', 'BTC-USD', 'LINK-USD', 'YFI-USD']

# Train-test split
train_size = 0.8  # 80% of data for training, 20% for testing
train_indices = int(len(prices) * train_size)

# Performance evaluation dictionary
evaluation_results = {}

# Train and evaluate models for each selected cryptocurrency
for crypto in selected_cryptos:
    print(f"Training and evaluating models for {crypto}...")
    
    # Get the price data for the current cryptocurrency
    crypto_prices = prices[crypto]
    X = np.arange(len(crypto_prices)).reshape(-1, 1)
    y = crypto_prices.values
    
    # Train-test split
    X_train, X_test = X[:train_indices], X[train_indices:]
    y_train, y_test = y[:train_indices], y[train_indices:]
    
    # ARIMA model
    arima_model = ARIMA(y_train, order=(5,1,0))
    arima_model_fit = arima_model.fit()
    arima_forecast = arima_model_fit.forecast(steps=len(X_test))[0]
    
    # LSTM model
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train, epochs=200, verbose=0)
    lstm_forecast = lstm_model.predict(X_test_lstm)
    
    # Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    rf_forecast = rf_regressor.predict(X_test)
    
    # Gradient Boosting Regressor
    gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_regressor.fit(X_train, y_train)
    gb_forecast = gb_regressor.predict(X_test)
    
    # Evaluate models
    arima_mae = mean_absolute_error(y_test, arima_forecast)
    lstm_mae = mean_absolute_error(y_test, lstm_forecast)
    rf_mae = mean_absolute_error(y_test, rf_forecast)
    gb_mae = mean_absolute_error(y_test, gb_forecast)
    
    evaluation_results[crypto] = {
        'ARIMA': arima_mae,
        'LSTM': lstm_mae,
        'RandomForest': rf_mae,
        'GradientBoosting': gb_mae
    }
    
    print(f"Evaluation results for {crypto}:")
    print("ARIMA MAE:", arima_mae)
    print("LSTM MAE:", lstm_mae)
    print("Random Forest MAE:", rf_mae)
    print("Gradient Boosting MAE:", gb_mae)
    print()

# Print overall evaluation results
print("Overall Evaluation Results:")
for crypto, results in evaluation_results.items():
    print(f"{crypto}:")
    for model, mae in results.items():
        print(f"{model}: MAE = {mae:.4f}")
    print()
