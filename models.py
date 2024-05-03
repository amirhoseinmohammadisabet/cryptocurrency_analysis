import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import numpy as np


def get_regression_model(algorithm):
    if algorithm == 'linear':
        return LinearRegression()
    elif algorithm == 'decision_tree':
        return DecisionTreeRegressor()
    elif algorithm == 'random_forest':
        return RandomForestRegressor()
    elif algorithm == 'gradient_boosting':
        return GradientBoostingRegressor()
    elif algorithm == 'knn':
        return KNeighborsRegressor(n_neighbors=5)
    elif algorithm == 'ann':
        return MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    else:
        raise ValueError('Invalid algorithm. Choose from "linear", "decision_tree", "random_forest", "gradient_boosting", "knn", or "ann".')
    
# Load hypothetical CSV file
df = pd.read_csv('Data/data.csv')

# special evaluation
def eval(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    r_squared_percentage = r_squared * 100
    rmse = np.sqrt(mse)
    normalized_rmse = rmse / (np.max(predictions) - np.min(predictions))
    error_rating = normalized_rmse
    accuracy = (1 - error_rating) * 100
    return{
        "accuracy": accuracy,
        "mse": mse,
        "r-squared": r_squared_percentage
    }

# Function to train a regression model and make predictions
def predict_currency_price(algorithm, base_currency, target_currency, price, market_cap, total_volume):
    # Extract relevant columns
    columns = [f"price_{base_currency}", f"market_cap_{base_currency}", f"total_volume_{base_currency}"]
    X = df[columns].values
    y = df[f"price_{target_currency}"].values

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Combine the user input with the existing data
    user_input = np.array([price, market_cap, total_volume])
    user_input_scaled = scaler.transform(user_input.reshape(1, -1))
    X_combined = np.vstack([X_scaled, user_input_scaled])

    # Train the selected regression model
    model = get_regression_model(algorithm)
    model.fit(X_combined[:-1], y)  # Train on all data except the last row (user input)

    # Make a prediction on the user input
    predicted_price = model.predict(X_combined[-1].reshape(1, -1))

    # Evaluate the model
    evals = eval(y, model.predict(X_combined[:-1]))
    print(f'Accuracy: {evals["accuracy"]:.2f}%')
    print(f'Mean Squared Error: {evals["mse"]}')
    print(f'R-squared: {evals["r-squared"]:.2f}%')

    # Use the mean of the target variable as a placeholder for target amount
    target_amount = np.mean(y)  
    
    # Convert ndarray to list before returning
    return {
        'mse': evals["mse"],
        'r_squared': evals["r-squared"],
        'predicted_price': predicted_price[0],
        'target_amount': target_amount  
    }


def choosing_four(palet):
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import matplotlib.pyplot as plt

    # Load the data
    data = pd.read_csv("Data/cryptocurrency_prices_cluster_transposed.csv", index_col=0)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_imputed)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(data_normalized)

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_normalized)

    # Apply LDA for visualization
    lda = LDA(n_components=2)
    data_lda = lda.fit_transform(data_normalized, clusters)

    # Choose one crypto from each cluster randomly
    chosen_cryptos = []
    for cluster_id in range(4):
        cluster_indices = np.where(clusters == cluster_id)[0]
        chosen_crypto = np.random.choice(cluster_indices)
        chosen_cryptos.append(chosen_crypto)

    if palet == 1:
        # Plot the scatter plot
        plt.figure(figsize=(12, 6))

        # PCA plot
        plt.subplot(1, 2, 1)
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
        for i, txt in enumerate(data.index):
            plt.annotate(txt, (data_pca[i, 0], data_pca[i, 1]), fontsize=8)
        plt.title('PCA')

        # LDA plot
        plt.subplot(1, 2, 2)
        plt.scatter(data_lda[:, 0], data_lda[:, 1], c=clusters, cmap='viridis', alpha=0.5)
        for i, txt in enumerate(data.index):
            plt.annotate(txt, (data_lda[i, 0], data_lda[i, 1]), fontsize=8)
        plt.title('LDA')

        plt.tight_layout()
        plt.show()

    # Display chosen cryptocurrencies
    x = []
    print("Chosen cryptocurrencies from each cluster:")
    for i, crypto_index in enumerate(chosen_cryptos):
        print(f"Cluster {i+1}: {data.index[crypto_index]}")
        x.append(data.index[crypto_index])
    return x



def top4_correlation():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load the CSV file into a DataFrame
    data = pd.read_csv('Data/cryptocurrency_prices_cluster.csv')

    # Assuming the first column contains the names of cryptocurrencies
    # and the remaining columns contain the prices over time
    cryptos = data.iloc[:, 0]
    prices = data.iloc[:, 1:]

    # Select the 4 cryptocurrencies you're interested in
    selected_cryptos = choosing_four(0)
    print(selected_cryptos)
    # selected_cryptos = ['MKR-USD', 'BTC-USD', 'LINK-USD', 'YFI-USD']

    # Select columns corresponding to the selected cryptocurrencies
    selected_prices = prices[selected_cryptos]

    # Compute the correlation matrix
    correlation_matrix = selected_prices.corr()

    # Find the top 4 highly correlated cryptocurrencies for each selected cryptocurrency
    top_positive_correlations = {}
    top_negative_correlations = {}

    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Selected Cryptocurrencies')
    plt.xlabel('Cryptocurrencies')
    plt.ylabel('Cryptocurrencies')
    plt.show()



def top4_eda():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the CSV file into a DataFrame
    data = pd.read_csv('Data/cryptocurrency_prices_cluster.csv')

    # Assuming the first column contains the names of cryptocurrencies
    # and the remaining columns contain the prices over time
    cryptos = data.iloc[:, 0]
    prices = data.iloc[:, 1:]

    # Select the 4 cryptocurrencies you're interested in
    selected_cryptos = choosing_four(0)

    # selected_cryptos = ['MKR-USD', 'BTC-USD', 'LINK-USD', 'YFI-USD']

    # Select columns corresponding to the selected cryptocurrencies
    selected_prices = prices[selected_cryptos]

    # EDA for each selected cryptocurrency
    fig, axes = plt.subplots(nrows=len(selected_cryptos), ncols=3, figsize=(16, 14))

    for i, crypto in enumerate(selected_cryptos):
        # Get the price data for the current cryptocurrency
        crypto_prices = selected_prices[crypto]
        
        # Plot the temporal structure
        axes[i, 0].plot(crypto_prices.index, crypto_prices.values, marker='o', linestyle='-')
        axes[i, 0].set_title(f'{crypto} Prices')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Price')
        axes[i, 0].grid(True)
        
        # Visualize the distribution of observations
        sns.histplot(crypto_prices, kde=True, bins=30, color='skyblue', ax=axes[i, 1])
        axes[i, 1].set_title(f'Distribution of {crypto} Prices')
        axes[i, 1].set_xlabel('Price')
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].grid(True)
        
        # Investigate the change in distribution over intervals
        first_half_prices = crypto_prices.iloc[:len(crypto_prices) // 2]
        second_half_prices = crypto_prices.iloc[len(crypto_prices) // 2:]
        
        sns.histplot(first_half_prices, kde=True, bins=30, color='blue', label='First Half', ax=axes[i, 2])
        sns.histplot(second_half_prices, kde=True, bins=30, color='orange', label='Second Half', ax=axes[i, 2])
        axes[i, 2].set_title(f'Distribution Comparison of {crypto} Prices')
        axes[i, 2].set_xlabel('Price')
        axes[i, 2].set_ylabel('Frequency')
        axes[i, 2].legend()
        axes[i, 2].grid(True)

    plt.tight_layout()
    plt.show()

def all():
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
    btc_data = yf.download('ENJ-USD', start='2022-01-01', end='2024-05-02')

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


def my_lstm():
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
