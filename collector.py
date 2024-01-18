import requests
import pandas as pd
from datetime import datetime, timedelta
from functools import reduce


def read_data_pd(csv_file_name):
    try:
        data = pd.read_csv(csv_file_name)
        return data
    except IOError:
        print("The file does not exist")
    except:
        print("an unexpected Error happened in file reader")

        

def get_price_data(days, coin):
    base_url = "https://api.coingecko.com/api/v3/coins/"
    vs_currency = "gbp"

    endpoint = f"{base_url}/{coin}/market_chart?vs_currency={vs_currency}&days={days}&interval=daily"
    response = requests.get(endpoint)

    if response.status_code == 200:
        data = response.json()
        prices = data["prices"]
        market_caps = data["market_caps"]
        total_volumes = data["total_volumes"]

        df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
        df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], unit="ms")

        df_market_caps = pd.DataFrame(market_caps, columns=["timestamp", "market_cap"])
        df_market_caps["timestamp"] = pd.to_datetime(df_market_caps["timestamp"], unit="ms")

        df_total_volumes = pd.DataFrame(total_volumes, columns=["timestamp", "total_volume"])
        df_total_volumes["timestamp"] = pd.to_datetime(df_total_volumes["timestamp"], unit="ms")

        # Merge the dataframes on the timestamp
        df = pd.merge(df_prices, df_market_caps, on="timestamp")
        df = pd.merge(df, df_total_volumes, on="timestamp")

        return df
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None



def save_to_csv(dataframe, file_path):
    try:
        dataframe.to_csv(file_path, index=False)
        print(f"Data has been successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")



# Example usage:
days = 20
btc_data = get_price_data(days, coin="bitcoin")
shiba_data = get_price_data(days, coin="shiba")
tron_data = get_price_data(days, coin="tron")
eth_data = get_price_data(days, coin="ethereum")

merged_data = pd.merge(btc_data, tron_data, on="timestamp", suffixes=('_btc', '_tron'))
print(btc_data)

# save_to_csv(merged_data, "Data/data.csv")
