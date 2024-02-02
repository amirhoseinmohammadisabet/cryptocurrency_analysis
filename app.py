from flask import Flask, render_template, request, jsonify
from models import predict_currency_price
import csv
import visualization as vis

app = Flask(__name__)

# Read data from CSV file into a list of dictionaries
with open('Data/data.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

@app.route('/')
def index():
    return render_template('index.html')

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

@app.route('/plot')
def plot():
    vis.generate_plot_html("btc")
    return render_template('plot.html')

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
