import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from prophet import Prophet
from sklearn.metrics import r2_score


# Step 1: Data Collection
btc_data = yf.download('BTC-USD', start='2023-01-01', end='2023-12-31')

# Step 2: Data Preprocessing
btc_data = btc_data[['Close']]  # Select only the 'Close' column
btc_data.dropna(inplace=True)    # Drop rows with missing values if any

# Split data into train and test sets
train_size = int(len(btc_data) * 0.8)  # 80% of data for training
test_size = len(btc_data) - train_size  # 20% of data for testing

train_data, test_data = btc_data.iloc[0:train_size], btc_data.iloc[train_size:]

# Convert data to numpy arrays
train_data = train_data.values
test_data = test_data.values

# Step 3: Model Training
# Linear Regression
X_train = np.arange(len(train_data)).reshape(-1, 1)
y_train = train_data.flatten()

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Ridge Regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

# Lasso Regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)

# ARIMA
arima_model = ARIMA(train_data, order=(5,1,0))
arima_model_fit = arima_model.fit()

# LSTM
# Reshape input to be [samples, time steps, features]
def create_dataset(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset)-time_steps-1):
        a = dataset[i:(i+time_steps), 0]
        X.append(a)
        Y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(Y)

time_steps = 10
X_train_lstm, y_train_lstm = create_dataset(train_data, time_steps)
X_test_lstm, y_test_lstm = create_dataset(test_data, time_steps)

X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], 1, X_train_lstm.shape[1]))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], 1, X_test_lstm.shape[1]))

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(1, time_steps)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=1, verbose=2)


# Prophet
prophet_train_data = pd.DataFrame({'ds': btc_data.index, 'y': btc_data['Close'].values})
prophet_model = Prophet()
prophet_model.fit(prophet_train_data)



# test
# Linear Regression
X_test_linear_reg = np.arange(train_size, len(btc_data)).reshape(-1, 1)

# Ridge Regression
X_test_ridge_reg = np.arange(train_size, len(btc_data)).reshape(-1, 1)

# Lasso Regression
X_test_lasso_reg = np.arange(train_size, len(btc_data)).reshape(-1, 1)


# Step 4: Evaluation


# Evaluate Linear Regression
linear_reg_train_preds = linear_reg.predict(X_train)
linear_reg_test_preds = linear_reg.predict(X_test_linear_reg)

linear_reg_train_mse = mean_squared_error(y_train, linear_reg_train_preds)
linear_reg_test_mse = mean_squared_error(test_data[-len(X_test_linear_reg):], linear_reg_test_preds)

print("Linear Regression Train MSE:", linear_reg_train_mse)
print("Linear Regression Test MSE:", linear_reg_test_mse)

# Evaluate Ridge Regression
ridge_reg_train_preds = ridge_reg.predict(X_train)
ridge_reg_test_preds = ridge_reg.predict(X_test_ridge_reg)

ridge_reg_train_mse = mean_squared_error(y_train, ridge_reg_train_preds)
ridge_reg_test_mse = mean_squared_error(test_data[-len(X_test_ridge_reg):], ridge_reg_test_preds)

print("Ridge Regression Train MSE:", ridge_reg_train_mse)
print("Ridge Regression Test MSE:", ridge_reg_test_mse)

# Evaluate Lasso Regression
lasso_reg_train_preds = lasso_reg.predict(X_train)
lasso_reg_test_preds = lasso_reg.predict(X_test_lasso_reg)

lasso_reg_train_mse = mean_squared_error(y_train, lasso_reg_train_preds)
lasso_reg_test_mse = mean_squared_error(test_data[-len(X_test_lasso_reg):], lasso_reg_test_preds)

print("Lasso Regression Train MSE:", lasso_reg_train_mse)
print("Lasso Regression Test MSE:", lasso_reg_test_mse)

# Evaluate ARIMA
arima_train_preds = arima_model_fit.predict(start=train_size, end=train_size+test_size-1, typ='levels')
arima_test_preds = arima_model_fit.forecast(test_size)

arima_train_mse = mean_squared_error(train_data[-len(arima_train_preds):], arima_train_preds)
arima_test_mse = mean_squared_error(test_data, arima_test_preds)

print("ARIMA Train MSE:", arima_train_mse)
print("ARIMA Test MSE:", arima_test_mse)

# Evaluate LSTM
lstm_train_preds = model_lstm.predict(X_train_lstm)
lstm_test_preds = model_lstm.predict(X_test_lstm)

lstm_train_mse = mean_squared_error(y_train_lstm, lstm_train_preds)
lstm_test_mse = mean_squared_error(y_test_lstm, lstm_test_preds)

print("LSTM Train MSE:", lstm_train_mse)
print("LSTM Test MSE:", lstm_test_mse)




# Evaluate Ridge Regression
ridge_reg_train_preds = ridge_reg.predict(X_train)
ridge_reg_test_preds = ridge_reg.predict(X_test_ridge_reg)
# Evaluate Lasso Regression
lasso_reg_train_preds = lasso_reg.predict(X_train)
lasso_reg_test_preds = lasso_reg.predict(X_test_lasso_reg)

# Step 5: Plotting
plt.figure(figsize=(14, 7))
plt.plot(btc_data.index, btc_data['Close'], label='Actual')

# Linear Regression
plt.plot(btc_data.index[train_size:], linear_reg.predict(np.arange(train_size, len(btc_data)).reshape(-1, 1)), label='Linear Regression')

# Ridge Regression
plt.plot(btc_data.index[train_size:], ridge_reg.predict(np.arange(train_size, len(btc_data)).reshape(-1, 1)), label='Ridge Regression')

# Lasso Regression
plt.plot(btc_data.index[train_size:], lasso_reg.predict(np.arange(train_size, len(btc_data)).reshape(-1, 1)), label='Lasso Regression')

# ARIMA
forecast_index = pd.date_range(start=btc_data.index[train_size], periods=test_size, freq='D')
plt.plot(forecast_index, arima_model_fit.forecast(test_size), label='ARIMA')

# LSTM
lstm_predictions = model_lstm.predict(X_test_lstm)
plt.plot(btc_data.index[train_size+time_steps+1:], lstm_predictions, label='LSTM')

# Prophet
future = prophet_model.make_future_dataframe(periods=test_size)
prophet_forecast = prophet_model.predict(future)
plt.plot(prophet_forecast['ds'][train_size:], prophet_forecast['yhat'][train_size:], label='Prophet')

plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


#plot each separately
# Plotting each model separately
plt.figure(figsize=(14, 7))

# Plot actual Bitcoin prices
plt.plot(btc_data.index, btc_data['Close'], label='Actual', color='black')

# Linear Regression
plt.plot(btc_data.index[train_size:], linear_reg.predict(np.arange(train_size, len(btc_data)).reshape(-1, 1)), label='Linear Regression', linestyle='--', color='blue')

# Ridge Regression
plt.plot(btc_data.index[train_size:], ridge_reg.predict(np.arange(train_size, len(btc_data)).reshape(-1, 1)), label='Ridge Regression', linestyle='--', color='green')

# Lasso Regression
plt.plot(btc_data.index[train_size:], lasso_reg.predict(np.arange(train_size, len(btc_data)).reshape(-1, 1)), label='Lasso Regression', linestyle='--', color='red')

plt.title('Bitcoin Price Prediction - Regression Models')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# ARIMA
plt.figure(figsize=(14, 7))
plt.plot(btc_data.index, btc_data['Close'], label='Actual', color='black')
forecast_index = pd.date_range(start=btc_data.index[train_size], periods=test_size, freq='D')
plt.plot(forecast_index, arima_model_fit.forecast(test_size), label='ARIMA', linestyle='--', color='blue')
plt.title('Bitcoin Price Prediction - ARIMA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# LSTM
plt.figure(figsize=(14, 7))
plt.plot(btc_data.index, btc_data['Close'], label='Actual', color='black')
plt.plot(btc_data.index[train_size+time_steps+1:], lstm_predictions, label='LSTM', linestyle='--', color='green')
plt.title('Bitcoin Price Prediction - LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Prophet
plt.figure(figsize=(14, 7))
plt.plot(btc_data.index, btc_data['Close'], label='Actual', color='black')
plt.plot(prophet_forecast['ds'][train_size:], prophet_forecast['yhat'][train_size:], label='Prophet', linestyle='--', color='red')
plt.title('Bitcoin Price Prediction - Prophet')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
