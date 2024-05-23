import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the pre-trained Random Forest model
rf_model = pickle.load(open('rf_model.sav', 'rb'))

# Load the input data
final_data = pd.read_csv('final_data1.csv')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    input_data = [float(x) for x in data.values()]
    output = rf_model.predict([input_data])[0]
    return jsonify({'Vitamin_D_Level': output})

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    output = rf_model.predict([data])[0]
    return render_template("home.html", prediction_text="Predicted Vitamin D Level: {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)