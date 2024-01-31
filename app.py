from flask import Flask, render_template, request, jsonify
import csv
import models

app = Flask(__name__)

# Read data from CSV file into a list of dictionaries
with open('Data/data.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/page1')
# def page1():
#     return render_template('page1.html')

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        # Assuming your form has input fields with names 'input1', 'input2', etc.
        input_data = [float(request.form.get('input1')),
                      float(request.form.get('input2')),
                      float(request.form.get('input3')),
                      float(request.form.get('input4')),
                      float(request.form.get('input5'))]
        
        try:
            # Use your models and functions to perform calculations
            model1, scaler1 = models.knn_model()
            result = models.predict_btc_price(input_data, model1, scaler1)

            return render_template('page1.html', result=result)
        except Exception as e:
            return render_template('page1.html', error=str(e))

    return render_template('page1.html')


@app.route('/filter')
def filter_results():
    name = request.args.get('name', '')

    filtered_data = [item for item in data if name.lower() in item['Name'].lower()]

    return jsonify(filtered_data)

if __name__ == '__main__':
    app.run(debug=True)
