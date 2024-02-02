from flask import Flask, render_template, request, jsonify
import csv
import test

app = Flask(__name__)

# Read data from CSV file into a list of dictionaries
with open('Data/data.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/output_plot')
def output_plot():
    test()
    return render_template('output_plot.html')

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
