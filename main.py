from flask import Flask, jsonify, request, abort
from flask_cors import CORS, cross_origin
import pickle
import sklearn

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

pkl_filename = 'model.pkl'
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_price():
    try:
        if request.method == 'POST':
            PRICE_WEIGHT = 0.6
            predict_data = request.json
            price = model.predict([[
                predict_data['processor_brand'],
                predict_data['processor_name'],
                predict_data['ram_gb'],
                predict_data['ssd'],
                predict_data['hdd'],
                predict_data['graphic_card_gb'],
                predict_data['Touchscreen']
            ]])[0]
            return jsonify({
                "price": int(price * PRICE_WEIGHT)
            })
    except:
        abort(400, 'Bad Request')

if __name__ == '__main__':
  app.run(port=5000)
