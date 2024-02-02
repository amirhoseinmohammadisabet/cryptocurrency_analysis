import pandas as pd
import matplotlib.pyplot as plt
import mpld3


# Read CSV file into a DataFrame
df = pd.read_csv('Data/data.csv')

# Convert 'timestamp' column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create subplots with added margins
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), gridspec_kw={'wspace': 0.3})

# Plot Bitcoin Price and Market Cap
axs[0, 0].plot(df['timestamp'], df['price_btc'], label='Bitcoin Price')
axs[0, 0].set_xlabel('Timestamp')
axs[0, 0].set_ylabel('Price (BTC)')
axs[0, 0].set_title('Bitcoin Price Over Time')
axs[0, 0].legend()

axs[0, 1].plot(df['timestamp'], df['market_cap_btc'], label='Bitcoin Market Cap', color='orange')
axs[0, 1].set_xlabel('Timestamp')
axs[0, 1].set_ylabel('Market Cap (BTC)')
axs[0, 1].set_title('Bitcoin Market Cap Over Time')
axs[0, 1].legend()

# Plot Tron Price and Market Cap
axs[1, 0].plot(df['timestamp'], df['price_tron'], label='Tron Price', color='green')
axs[1, 0].set_xlabel('Timestamp')
axs[1, 0].set_ylabel('Price (TRON)')
axs[1, 0].set_title('Tron Price Over Time')
axs[1, 0].legend()

axs[1, 1].plot(df['timestamp'], df['market_cap_tron'], label='Tron Market Cap', color='red')
axs[1, 1].set_xlabel('Timestamp')
axs[1, 1].set_ylabel('Market Cap (TRON)')
axs[1, 1].set_title('Tron Market Cap Over Time')
axs[1, 1].legend()

# Manually adjust layout parameters
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

# Convert to HTML
html_plot = mpld3.fig_to_html(plt.gcf())

# Save the HTML plot to a file or serve it using a web framework
with open('templates/output_plot.html', 'w') as f:
    f.write(html_plot)


# Show the plot
# plt.show()
