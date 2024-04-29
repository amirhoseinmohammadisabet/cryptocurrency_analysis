from flask import Flask, render_template, request, jsonify
from models import predict_currency_price
import csv
import visualization as vis
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


app = Flask(__name__)

# Read data from CSV file into a list of dictionaries
with open('Data/data.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Function to perform forecasting using Prophet
def forecast(data):
    # Prepare data for Prophet
    df = pd.DataFrame()
    df['ds'] = data.index
    df['y'] = data['Close'].values

    # Create Prophet model and fit the data
    model = Prophet()
    model.fit(df)

    # Make future predictions
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast

def get_cryptocurrency_data(symbol, period):
    # Fetch data from Yahoo Finance
    data = yf.download(symbol, period=period)
    return data

def generate_candlestick_plot(data, symbol, period):
    candlestick = go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'])

    layout = go.Layout(title=f'{symbol} Candlestick Chart ({period})',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Price'))

    fig = go.Figure(data=[candlestick], layout=layout)
    return fig.to_html(full_html=False)

def generate_line_chart(data, symbol, period):
    trace = go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price')
    
    layout = go.Layout(title=f'{symbol} Line Chart ({period})',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Price'))

    fig = go.Figure(data=[trace], layout=layout)
    return fig.to_html(full_html=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signal')
def signal():
    return render_template('signal.html')

@app.route('/result', methods=['POST'])
def result():
    # Get form inputs
    cryptocurrency = request.form['cryptocurrency']
    period = request.form['period']
    buy_threshold = float(request.form['buy_threshold'])
    sell_threshold = float(request.form['sell_threshold'])
    initial_balance = float(request.form['initial_balance'])

    # Fetch historical cryptocurrency price data using Yahoo Finance
    data = yf.download(cryptocurrency.upper() + '-USD', start='2023-01-01', end='2023-12-31')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    data = data[['ds', 'y']]

    # Train a Prophet model
    model = Prophet()
    model.fit(data)

    # Forecast future prices
    future = model.make_future_dataframe(periods=int(period))
    forecast = model.predict(future)

    # Generate buy and sell signals
    balance = initial_balance
    buy_price = 0
    results = []

    for index, row in forecast.iterrows():
        if row['yhat_upper'] is not None and row['yhat_lower'] is not None:
            upper_price = row['yhat_upper']
            lower_price = row['yhat_lower']
            if row['yhat'] - lower_price > buy_threshold and buy_price == 0:
                buy_price = row['yhat']
                results.append((row['ds'], 'Buy', buy_price, balance))
            elif row['yhat'] - upper_price < sell_threshold and buy_price > 0:
                profit = row['yhat'] - buy_price
                balance += profit
                results.append((row['ds'], 'Sell', row['yhat'], balance))
                buy_price = 0

    return render_template('result.html', results=results)

@app.route('/visual')
def visual():
    return render_template('visual.html')


@app.route('/show')
def show():
    return render_template('show.html')

@app.route('/plot', methods=['POST'])
def plot():
    # Get form data
    symbol = request.form['cryptocurrency']
    period = request.form['period']
    chart_type = request.form['chart_type']

    # Get cryptocurrency data
    data = get_cryptocurrency_data(symbol, period)

    # Generate plot based on chart type
    if chart_type == 'candlestick':
        plot_html = generate_candlestick_plot(data, symbol, period)
    elif chart_type == 'line':
        plot_html = generate_line_chart(data, symbol, period)
    else:
        plot_html = '<p>Invalid chart type selected</p>'

    return plot_html


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    algorithms = ['linear', 'decision_tree', 'random_forest', 'gradient_boosting', 'knn', 'ann']
    currencies = ['btc', 'eth', 'tron']  # Add more currencies as needed

    if request.method == 'POST':
        algorithm = request.form.get('algorithm')
        base_currency = request.form.get('base_currency')
        target_currency = request.form.get('target_currency')
        price = float(request.form.get('price'))
        market_cap = float(request.form.get('market_cap'))
        total_volume = float(request.form.get('total_volume'))

        result = predict_currency_price(algorithm, base_currency, target_currency, price, market_cap, total_volume)

        # Convert the result to JSON before returning
        return jsonify(result=result, algorithms=algorithms, currencies=currencies)

    return render_template('prediction.html', algorithms=algorithms, currencies=currencies)

def create_dataset(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset)-time_steps-1):
        a = dataset[i:(i+time_steps), 0]
        X.append(a)
        Y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(Y)


@app.route('/page1', methods=['GET', 'POST'])
def page1():
    return render_template('page1.html')


@app.route('/filter')
def filter_results():
    name = request.args.get('name', '')

    filtered_data = [item for item in data if name.lower() in item['price_btc'].lower()]
    
    return jsonify(filtered_data)

if __name__ == '__main__':
    app.run(debug=True)
