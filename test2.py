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

# Load hypothetical CSV file
df = pd.read_csv('Data/data.csv')

# Function to train a regression model and make predictions
def predict_currency_price(algorithm, base_currency, target_currency):
    # Extract relevant columns
    columns = [f"price_{base_currency}", f"market_cap_{base_currency}", f"total_volume_{base_currency}"]
    X = df[columns].values
    y = df[f"price_{target_currency}"].values

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split scaled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Choose the regression model based on the selected algorithm
    if algorithm == 'linear':
        model = LinearRegression()
    elif algorithm == 'decision_tree':
        model = DecisionTreeRegressor()
    elif algorithm == 'random_forest':
        model = RandomForestRegressor()
    elif algorithm == 'gradient_boosting':
        model = GradientBoostingRegressor()
    elif algorithm == 'knn':
        # Perform basic hyperparameter tuning for KNN
        model = KNeighborsRegressor(n_neighbors=5)
    elif algorithm == 'ann':
        # Perform basic hyperparameter tuning for ANN
        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    else:
        raise ValueError('Invalid algorithm. Choose from "linear", "decision_tree", "random_forest", "gradient_boosting", "knn", or "ann".')

    # Train the selected regression model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)

    # Convert R-squared to a percentage
    r_squared_percentage = r_squared * 100

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r_squared_percentage:.2f}%')

    # Ask user for input to make a prediction
    user_input = np.array([float(x) for x in input(f'Enter the {base_currency} values (price, market cap, total volume): ').split(',')])

    # Scale user input and make a prediction using the trained model
    user_input_scaled = scaler.transform(user_input.reshape(1, -1))
    predicted_price = model.predict(user_input_scaled)
    print(f'Predicted {target_currency} price using {algorithm} regression: {predicted_price[0]}')

# Example usage
predict_currency_price("linear", "btc", "tron")
predict_currency_price("decision_tree", "btc", "tron")
predict_currency_price("random_forest", "btc", "tron")
predict_currency_price("gradient_boosting", "btc", "tron")
predict_currency_price("knn", "btc", "tron")
predict_currency_price("ann", "btc", "tron")
