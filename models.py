import pandas as pd
import collector as col
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import numpy as np



data = col.read_data_pd("Data/data.csv")

def prepro(data):
    features = ['price_tron', 'market_cap_tron', 'total_volume_tron','market_cap_btc','total_volume_btc']
    X = data[features]
    y = data['price_btc']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled,X_test_scaled,y_train,y_test

def knn_eval():
    X_train,X_test,y_train,y_test = prepro(data)
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

def ann_eval():
    X_train, X_test, y_train, y_test = prepro(data)  # Assuming prepro is your preprocessing function
    # Create and configure the ANN model
    ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    ann_model.fit(X_train, y_train)
    y_pred = ann_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    normalized_rmse = rmse / (np.max(y_pred) - np.min(y_pred))
    error_rating = normalized_rmse
    print(f'Error Rating: {error_rating:.2f}')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    
    return ann_model,rmse,normalized_rmse  # You can return the trained model if needed for further use

def predict_btc_price(input_data, knn_model, scaler):
    # Prepare input data for prediction
    input_data_scaled = scaler.transform([input_data])

    # Make prediction for the specific date
    btc_price_prediction = knn_model.predict(input_data_scaled)
    print(f'Predicted BTC Price: {btc_price_prediction[0]}')

# ann_model,rmse,normalized_rmse = ann_eval()
# scaler = StandardScaler()
# input_data = scaler.transform([[0.0812, 7194154492, 235471805, 46621242074,4118765210]])
# input_data = [0.0812, 7194154492, 235471805, 46621242074, 4118765210]

# price_prediction = ann_model.predict(input_data)
# print(price_prediction)



# ann_model,rmse,normalized_rmse = ann_eval()

# input_data_for_prediction = [0.0812, 7194154492, 235471805, 46621242074, 4118765210]

# predict_btc_price(input_data_for_prediction, ann_model, scaler)