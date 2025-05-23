import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json

app = Flask(__name__)
CORS(app)

# Load trained model and columns
with open(r'G:\python_pycharm\machine learning\project\BangloreHomePrices\model\bangalore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)
with open(r'G:\python_pycharm\machine learning\project\BangloreHomePrices\server\artifacts\columns (1).json', 'r') as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]

def get_estimated_price(location, sqft, bhk, bath):
    x = np.zeros(len(data_columns))
    x[data_columns.index('total_sqft')] = sqft
    x[data_columns.index('bath')] = bath
    x[data_columns.index('bhk')] = bhk
    if location.lower() in data_columns:
        x[data_columns.index(location.lower())] = 1
    return round(model.predict([x])[0], 2)

def get_location_names():
    return locations

@app.route('/get_location_names', methods=['GET'])
def get_location_names_endpoint():
    response = jsonify({'locations': get_location_names()})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    data = request.get_json()
    estimated_price = get_estimated_price(data['location'], float(data['total_sqft']), int(data['bhk']), int(data['bath']))
    response = jsonify({
        'estimated_price': estimated_price
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    app.run(debug=True)
