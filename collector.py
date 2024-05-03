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


def tosave():
    # Example usage:
    days = 500
    btc_data = get_price_data(days, coin="bitcoin")
    # shiba_data = get_price_data(days, coin="shiba")
    tron_data = get_price_data(days, coin="tron")
    # eth_data = get_price_data(days, coin="ethereum")
    merged_data = pd.merge(btc_data, tron_data, on="timestamp", suffixes=('_btc', '_tron'))
    # print(btc_data)

    save_to_csv(merged_data, "Data/data1.csv")

def crypto_for_clustering():
    import yfinance as yf
    import pandas as pd

    # List of 30 most important cryptocurrencies
    cryptocurrencies = [
        "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD",
        "LTC-USD", "ALGO-USD", "MATIC-USD", "FIL-USD", "TRX-USD",
        "XTZ-USD", "VET-USD", "EGLD-USD", "XLM-USD", "AAVE-USD",
        "SUSHI-USD", "CAKE-USD", "THETA-USD", "EOS-USD", "CHZ-USD",
        "DOGE-USD", "ZEC-USD", "AAVE-USD", "AVAX-USD", "ATOM-USD",
        "NEO-USD", "ATOM-USD", "XMR-USD", "SOL-USD", "DOT-USD",
        "TRX-USD", "SHIB-USD","BTC-USD","YFI-USD",
        "UNI-USD", "MKR-USD", "LINK-USD", "SNX-USD",
        "HT-USD", "FTT-USD", "ENJ-USD", "MANA-USD", 
        "IOST-USD", "ICX-USD", "REN-USD", "BTT-USD",
        "LRC-USD", "ZRX-USD", "OMG-USD", "1INCH-USD", "WAVES-USD",
        "SC-USD", "BCH-USD", "QTUM-USD", "DASH-USD", "RVN-USD",
        "HBAR-USD", "XEM-USD", "WTC-USD", "BAT-USD", "XDC-USD",
        "ANKR-USD", "HOT-USD", "ETC-USD", "CRV-USD", "DOGE-USD",
        "KSM-USD", "SNX-USD", "NEAR-USD", "REP-USD",
        "REN-USD", "ZIL-USD", "RUNE-USD", "MIR-USD", "AAVE-USD",
        "SKL-USD", "ALGO-USD", "MANA-USD", "STX-USD", "ICP-USD",
        "HBAR-USD", "NU-USD", "BNT-USD", "BTS-USD", "SXP-USD",
        "ARDR-USD", "CELO-USD", "LSK-USD", "KNC-USD", "GNO-USD",
        "RLC-USD", "BAL-USD", "FORTH-USD", "ETN-USD", "DCR-USD"
    ]

    # Fetch historical data for each cryptocurrency
    data = {}
    for cryptocurrency in cryptocurrencies:
        df = yf.download(cryptocurrency, start="2023-02-06", end="2024-02-06")
        data[cryptocurrency] = df['Close']

    # Convert data dictionary to DataFrame
    df = pd.DataFrame(data)
    df.to_csv('Data/cryptocurrency_prices_cluster.csv')

    df = df.T

    # Save data to CSV file
    df.to_csv('Data/cryptocurrency_prices_cluster_transposed.csv')
    return df
