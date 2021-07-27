# -*- coding: utf-8 -*-
"""
Crée le 15 juillet 2021

@author: Cécile Guillot
"""

# importing librairies
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import  train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

# loading the dataset
data = pd.read_csv("datasets/benchmark_total.csv")

#######################
######ENERGY USE#######
#######################

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42, stratify=data["PrimaryPropertyType"])

y_train = train_set["SiteEnergyUse(kBtu)"].values
y_test = test_set["SiteEnergyUse(kBtu)"].values
X_train = train_set.drop(["SiteEnergyUse(kBtu)", "TotalGHGEmissions", "ENERGYSTARScore", "Age", "YearBuilt", "Neighborhood", "NumberofBuildings", "NumberofFloors", "Latitude", "Longitude"], axis=1)
X_test = test_set.drop(["SiteEnergyUse(kBtu)", "TotalGHGEmissions", "ENERGYSTARScore", "Age", "YearBuilt", "Neighborhood", "NumberofBuildings", "NumberofFloors","Latitude", "Longitude"], axis=1)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Preprocessing of categorical and numerical variables
cat_var = ["PrimaryPropertyType", "NbofFloors", "NbofBuildings", "HasParking", "Clusters", "Bins_Age"]
num_var = ["PropertyGFATotal", "degreeDaysH", "degreeDaysC"]

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
xgboost_energy = xgb.XGBRegressor(objective="reg:squarederror", booster="gbtree", 
                                    gamma=1, learning_rate=0.15, max_depth=12, sampling_method="uniform",
                                    n_estimators=400, tree_method="hist", random_state=42, n_jobs=-1)
xgboost_energy.fit(X_train, y_train)

# Final pipeline
full_pipeline_energy = Pipeline([
    ("preprocessing", preprocessor),
    ("model", xgboost_energy)
])
joblib.dump(full_pipeline_energy, 'API_deploiement/models/model_prediction_energy.pkl')

#######################
#####GHG EMISSIONS#####
#######################

# Preparation of the different datasets
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42, stratify=data["PrimaryPropertyType"])

y_train = train_set["TotalGHGEmissions"].values
y_test = test_set["TotalGHGEmissions"].values
X_train = train_set.drop(["SiteEnergyUse(kBtu)", "TotalGHGEmissions", "ENERGYSTARScore", "YearBuilt", "Age", "Neighborhood", "NumberofBuildings", "NumberofFloors", "Latitude", "Longitude"], axis=1)
X_test = test_set.drop(["SiteEnergyUse(kBtu)", "TotalGHGEmissions", "ENERGYSTARScore", "YearBuilt", "Age", 'Neighborhood', "NumberofBuildings", "NumberofFloors", "Latitude", "Longitude"], axis=1)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Preprocessing of the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Preparation of the 2nd model
xgboost_co2 = xgb.XGBRegressor(objective="reg:squarederror", booster="gbtree", 
                               gamma=1, learning_rate=0.1, max_depth=12, sampling_method="uniform",
                               n_estimators = 400, tree_method="hist", random_state=42, n_jobs=-1)

xgboost_co2.fit(X_train, y_train)

# Final pipeline
full_pipeline_co2 = Pipeline([
    ("preprocessing", preprocessor),
    ("model", xgboost_co2)
])
joblib.dump(full_pipeline_co2, "API_deploiement/models/model_prediction_co2.pkl")

# Test
new_building = pd.DataFrame({"PrimaryPropertyType": "Hospital",
                             "PropertyGFATotal": 56700,
                             "NbofFloors": "2+f",
                             "NbofBuildings" : "2+b",
                             "Bins_Age" : "Recent",
                             "degreeDaysH" : 4438,
                             "degreeDaysC" : 247,
                             "HasParking" : "Yes",
                             "Clusters" : "2"}, 
                             index=[1])
print(new_building)
print("Estimation de la consommation d'énergie :" + str((full_pipeline_energy.predict(new_building)).round(2)) + " kBtu")
print("Estimation des émissions de CO2 :" + str((full_pipeline_co2.predict(new_building)).round(2)) + " MetricTonsCO2e")