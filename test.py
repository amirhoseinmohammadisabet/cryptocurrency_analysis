import requests
import pandas as pd
from datetime import datetime, timedelta

# Function to get historical price data for Bitcoin from CoinGecko API
def get_btc_price_data(days):
    base_url = "https://api.coingecko.com/api/v3/coins/tron/market_chart"
    vs_currency = "gbp"

    # Calculate the start date based on the number of days to go back
    # start_date = datetime.utcnow() - timedelta(days=days)

    # API endpoint for historical price data
    endpoint = f"{base_url}?vs_currency={vs_currency}&days={days}&interval=daily"

    # Make the API request
    response = requests.get(endpoint)

    if response.status_code == 200:
        data = response.json()
        prices = data["prices","market_caps","total_volumes"]

        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    else:
        print(f"Error: {response.status_code}")
        print(response.text)  # Print the response text for further debugging
        return None

# Example usage:
btc_data = get_btc_price_data(days=30)
print(btc_data)
