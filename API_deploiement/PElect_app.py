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
    prediction_co2 = model_co2.predict(df)
    output_co2 = round(prediction_co2[0], 2)
    prediction_energy = model_energy.predict(df)
    output_energy = round(prediction_energy[0], 0)
    return render_template('index.html', prediction_text="Site Energy Use:  {} kBtu & GHG Emissions: {} Metrics Tons CO2.".format(output_energy, output_co2))

if __name__ == "__main__":
    model_co2 = joblib.load("./models/model_prediction_co2.pkl")
    model_energy = joblib.load("./models/model_prediction_energy.pkl")
    print("Models loaded")
    app.run(host="localhost", port=5000, debug=True)