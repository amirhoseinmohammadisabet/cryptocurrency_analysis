import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from flask import Flask, render_template, request, jsonify


def output_plot_html(currency):
    # Load CSV data into a DataFrame
    df = pd.read_csv('Data/data.csv', parse_dates=['timestamp'])

    # Plotting using Plotly
    fig = go.Figure()
    if currency == "btc":
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price_btc'], mode='lines', name='BTC Price',
                                hovertext=df['price_btc'].apply(lambda x: f'Price: {x:.2f} BTC'),
                                line=dict(color='purple')))  # Set the line color
    elif currency == "trx":
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price_tron'], mode='lines', name='TRX Price',
                                hovertext=df['price_tron'].apply(lambda x: f'Price: {x:.2f} TRX'),
                                line=dict(color='orange')))  # Set the line color
    else: print("give it a currency")
    # Add rangeslider for scrolling through date ranges
    fig.update_layout(
        title= (currency.upper)+' Price Over Time',
        xaxis_title='Date',
        yaxis_title=(currency.upper)+' Price',
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        paper_bgcolor='white',  # Set the background color
        plot_bgcolor='antiquewhite'        # Set the plot area color
    )

    # Save the plot to an HTML file
    pyo.plot(fig, filename='plot.html')

output_plot_html("trx")

# app = Flask(__name__)
# @app.route('/visualization', methods=['GET', 'POST'])
# def prediction():
#     algorithms = ['linear', 'decision_tree', 'random_forest', 'gradient_boosting', 'knn', 'ann']
#     currencies = ['btc', 'eth', 'tron']  # Add more currencies as needed

#     if request.method == 'POST':
#         currency = request.form.get('base_currency')

#         result = output_plot_html(currency)

#         # Convert the result to JSON before returning
#         return jsonify(result=result)

#     return render_template('template/visualization.html', currencies=currencies)