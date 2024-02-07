import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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

# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data_normalized[:, 0], data_normalized[:, 1], c=clusters, cmap='viridis', alpha=0.5)

# Annotate points with cryptocurrency names
for i, txt in enumerate(data.index):
    plt.annotate(txt, (data_normalized[i, 0], data_normalized[i, 1]), fontsize=8)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering without PCA or LDA')

plt.show()

# Choose one crypto from each cluster randomly
chosen_cryptos = []
for cluster_id in range(4):
    cluster_indices = np.where(clusters == cluster_id)[0]
    chosen_crypto = np.random.choice(cluster_indices)
    chosen_cryptos.append(chosen_crypto)

# Display chosen cryptocurrencies
print("Chosen cryptocurrencies from each cluster:")
for i, crypto_index in enumerate(chosen_cryptos):
    print(f"Cluster {i+1}: {data.index[crypto_index]}")
