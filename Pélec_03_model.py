# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 19:50:36 2021

@author: cecil
"""

# importing librairies
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import  train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# loading the dataset
data = pd.read_csv("datasets/benchmark_total.csv")

# Separation of the features and the two targets
y_energy = data["SiteEnergyUse(kBtu)"].values
y_ghg = data["TotalGHGEmissions"].values
X = data.drop(["SiteEnergyUse(kBtu)", "TotalGHGEmissions", "ENERGYSTARScore"], axis=1)

#######################
######ENERGY USE#######
#######################

# Preparation of the different datasets
X_train, X_test, y_train, y_test = train_test_split(X, y_energy, 
                                                    test_size=0.2, random_state=42)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


# Preprocessing of categorical and numerical variables
cat_var = ["PrimaryPropertyType", "Neighborhood"]
num_var = ["YearBuilt", "NumberofBuildings", "NumberofFloors", "PropertyGFATotal"]

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy="median", fill_value="missing")),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipe, cat_var),
    ('num', num_pipe, num_var)
])
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# Preparation of the model
forest_reg_energy = RandomForestRegressor(n_estimators=350, min_samples_leaf=3, max_depth=4,
                                   min_samples_split=2, max_features="auto", random_state=42)
forest_reg_energy.fit(X_train, y_train)

# Final pipeline
full_pipeline_energy = Pipeline([
    ("preprocessing", preprocessor),
    ("model", forest_reg_energy)
])
joblib.dump(full_pipeline_energy, 'model_prediction_energy.pkl')

#######################
#####GHG EMISSIONS#####
#######################

# Preparation of the different datasets
X_train, X_test, y_train, y_test = train_test_split(X, y_ghg, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Preprocessing of the data
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)

# Preparation of the 2nd model
forest_reg_co2 = RandomForestRegressor(n_estimators=460, min_samples_leaf=2, max_depth=4,
                                       min_samples_split=2, max_features="auto", random_state=42)
forest_reg_co2.fit(X_train, y_train)

# Final pipeline
full_pipeline_co2 = Pipeline([
    ("preprocessing", preprocessor),
    ("model", forest_reg_co2)
])
joblib.dump(full_pipeline_co2, "model_prediction_co2.pkl")

# Test
new_building = pd.DataFrame({"PrimaryPropertyType": "Hospital",
                             "Neighborhood" : "Delridge",
                             "YearBuilt" : 2016,
                             "NumberofBuildings": 5,
                             "NumberofFloors": 5, 
                             "PropertyGFATotal": 56700}, index=[1])
print(new_building)
print("Estimation de la consommation d'énergie :" + str((full_pipeline_energy.predict(new_building)).round(2)) + " kBtu")
print("Estimation des émissions de CO2 :" + str((full_pipeline_co2.predict(new_building)).round(2)) + " MetricTonsCO2e")