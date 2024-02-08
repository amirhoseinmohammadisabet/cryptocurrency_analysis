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


def choosing_four():
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
    print("Chosen cryptocurrencies from each cluster:")
    for i, crypto_index in enumerate(chosen_cryptos):
        print(f"Cluster {i+1}: {data.index[crypto_index]}")




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
    selected_cryptos = ['MKR-USD', 'BTC-USD', 'LINK-USD', 'YFI-USD']

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
    selected_cryptos = ['MKR-USD', 'BTC-USD', 'LINK-USD', 'YFI-USD']

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