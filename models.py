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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def eval(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    normalized_rmse = rmse / (np.max(y_pred) - np.min(y_pred))
    error_rating = normalized_rmse
    accuracy = (1 - error_rating) * 100
    print(f'Accuracy: {accuracy:.2f}%')

def knn_model():
    X_train, X_test, y_train, y_test, scaler = prepro(data)
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    eval(y_test, y_pred)
    return knn_model, scaler

def ann_model():
    X_train, X_test, y_train, y_test, scaler = prepro(data)
    ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=5000, random_state=42)
    ann_model.fit(X_train, y_train)
    y_pred = ann_model.predict(X_test)
    eval(y_test, y_pred)
    return ann_model, scaler

def predict_btc_price(input_data, model, scaler):
    input_data_scaled = scaler.transform([input_data])
    btc_price_prediction = model.predict(input_data_scaled)
    print(f'Predicted BTC Price: {btc_price_prediction[0]}')

    