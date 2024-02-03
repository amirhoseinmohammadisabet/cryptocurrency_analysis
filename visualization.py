import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import random

def generate_plot_html(currency):
    # Define a mapping dictionary for currency symbols and corresponding columns
    currency_mapping = {
        "btc": "price_btc",
        "trx": "price_tron",
        # Add more currencies as needed
    }

    # Check if the given currency is in the mapping dictionary
    if currency in currency_mapping:
        # Load CSV data into a DataFrame
        df = pd.read_csv('Data/data.csv', parse_dates=['timestamp'])

        # Plotting using Plotly
        fig = go.Figure()

        # Use the mapping dictionary to dynamically select the column
        column_name = currency_mapping[currency]

        # Generate a random color for the plot
        random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[column_name],
            mode='lines',
            name=f'{currency.upper()} Price',
            hovertext=df[column_name].apply(lambda x: f'Price: {x:.2f} {currency.upper()}'),
            line=dict(color=random_color) 
        ))

        # Add rangeslider for scrolling through date ranges
        fig.update_layout(
            title=f"{currency.upper()} Price Over Time",
            xaxis_title='Date',
            yaxis_title=f"{currency.upper()} Price",
            xaxis=dict(rangeslider=dict(visible=True), type='date'),
            paper_bgcolor='white',  
            plot_bgcolor='antiquewhite' 
        )

        # Save the plot to an HTML file
        plot_filename = f"templates/plot.html"
        pyo.plot(fig, filename=plot_filename,auto_open=False)

        return plot_filename
    else:
        print("Give it a valid currency")


