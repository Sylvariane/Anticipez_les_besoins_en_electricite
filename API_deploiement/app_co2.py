# -*- coding: utf-8 -*-
"""
Crée le 18/07/2021

@author: Cécile Guillot
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import joblib
import traceback
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods= ['POST'])
def predict():
    d = None
    if request.method == 'POST':
        print('POST received')
        d = request.form.to_dict()
    else:
        print('GET received')
        d = request.args.to_dict()
    print(d)
    print(d.keys())
    print(d.values())

    print("Dataframe format required for Machine Learning prediction")
    df = pd.DataFrame([d.values()], columns=d.keys())
    print(df)
    prediction = model.predict(df)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text="GHG Emissions should be {} Metrics Tons CO2".format(output))

if __name__ == "__main__":
    model = joblib.load("./models/model_prediction_co2s.pkl")
    print("Model loaded")
    app.run(host="localhost", port=5000, debug=True)