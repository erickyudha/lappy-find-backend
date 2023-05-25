from flask import Flask, jsonify, request, abort
import pickle
import sklearn

app = Flask(__name__)

pkl_filename = 'model.pkl'
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

@app.route("/predict", methods=['GET'])
def predict_price():
    try:
        if request.method == 'GET':
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
