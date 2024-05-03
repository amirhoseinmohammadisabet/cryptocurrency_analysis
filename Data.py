import requests
import csv
from datetime import datetime, timedelta

def get_crypto_data(symbol, start_date, end_date):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": start_date,
        "to": end_date
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    market_caps = data['market_caps']
    total_volumes = data['total_volumes']
    return prices, market_caps, total_volumes

def save_to_csv(prices, market_caps, total_volumes, filename, symbol):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Date', 'Price', 'Market Cap', 'Total Volume']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(prices)):
            writer.writerow({
                'Date': datetime.fromtimestamp(prices[i][0] / 1000).strftime('%Y-%m-%d'),
                'Price': prices[i][1],
                'Market Cap': market_caps[i][1],
                'Total Volume': total_volumes[i][1]
            })

# List of cryptocurrencies
symbols = ["bitcoin", "maker", "yearn-finance", "enjincoin"]

# Calculate start and end dates (one year ago from today)
end_date = int(datetime.now().timestamp())
start_date = end_date - (365 * 24 * 60 * 60)  # One year in seconds

for symbol in symbols:
    # Fetch data
    prices, market_caps, total_volumes = get_crypto_data(symbol, start_date, end_date)
    # Save data to CSV
    save_to_csv(prices, market_caps, total_volumes, f'{symbol}_historical_data.csv', symbol)

print("CSV files created successfully.")
