import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Step 1: Download BTC close price data
btc_data = yf.download('ENJ-USD', start='2022-01-01', end='2024-05-02')

# Step 2: Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
btc_close = btc_data['Close'].values.reshape(-1, 1)
btc_close_scaled = scaler.fit_transform(btc_close)

# Split data into training and testing sets
train_size = int(len(btc_close_scaled) * 0.8)
test_size = len(btc_close_scaled) - train_size
train_data, test_data = btc_close_scaled[0:train_size, :], btc_close_scaled[train_size:len(btc_close_scaled), :]

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 3: Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Step 4: Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=1)

# Step 5: Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Step 6: Plot the results
plt.figure(figsize=(15, 7))

# Plot actual train prices
plt.plot(np.arange(0, len(train_predict)), y_train.flatten(), 'g', label="Actual Train Price")

# Calculate the range for the actual test prices
test_range_actual = np.arange(len(train_predict), len(train_predict) + len(y_test.flatten()))

# Plot actual test prices
plt.plot(test_range_actual, y_test.flatten(), 'r', label="Actual Test Price")

# Plot predicted train prices
plt.plot(np.arange(0, len(train_predict)), train_predict.flatten(), 'b', label="Predicted Train Price")

# Calculate the range for the predicted test prices
test_range_predicted = np.arange(len(train_predict), len(train_predict) + len(test_predict.flatten()))

# Plot predicted test prices
plt.plot(test_range_predicted, test_predict.flatten(), 'm', label="Predicted Test Price")

plt.xlabel('Time')
plt.ylabel('BTC Price')
plt.legend()
plt.show()
