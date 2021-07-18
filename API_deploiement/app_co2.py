# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:06:28 2021

@author: cecil
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 15:15:15 2021

@author: cecil
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import joblib
import traceback
import pandas as pd

app = Flask(__name__)

@app.route('/upload')
def upload_file():
    return render_template("upload.html")

@app.route("/uploader", methods= ['POST'])
def uploader_file():
    f = request.files["upload_file"]
    f.save(secure_filename(f.filename))
    if final_model_co2:
        try:
            df = pd.read_csv(f.filename)
            query= df[["PrimaryPropertyType", "Neighborhood", 
                       "YearBuilt", "NumberofBuildings", 
                       "NumberofFloors", "PropertyGFATotal"]]
            print(query)
            prediction = list(final_model_co2.predict(query))
            print(list(final_model_co2.predict(query)))
            
            dictionary = dict(zip(df.BuildingName, prediction))
            
            return jsonify(str(dictionary))
        except:
            return jsonify({'trace' : traceback.format_exc()})
        else:
            print("Train the model first")
            return("No model here to use")

if __name__ == "__main__":
    final_model_co2 = joblib.load("./models/model_prediction_co2.pkl")
    print("Model loaded")
    app.run(host="localhost", port=5000, debug=True)