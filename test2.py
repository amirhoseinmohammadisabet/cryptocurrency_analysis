from flask import Flask, render_template, request, jsonify
from models import predict_currency_price


app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)
