from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from ml_training.nn_model import NNModel  # noqa: F401


model = joblib.load("trained_model/neural_network.pkl")
countries = joblib.load("trained_model/countries.pkl")
country_encoder = joblib.load("trained_model/country_encoder.pkl")

# Initialize API
app = FastAPI(title="Energy Forecast API")

# Returns the predicted TOE_HAB consumption of a country for a specified year
@app.get("/predict")
def predict(country:str, year: int):
    c = country_encoder.transform(np.array([country]))[0]
    X = pd.DataFrame.from_dict({"country_encoded": [c], "year": [year]}).astype("float32")
    prediction = model.predict(X)
    return {"country": country, "year": year, "TOE_HAB": float(prediction.values[0])}

# Returns the predicted TOE_HAB consumption of all EU countries for a specified year
@app.get("/predict-year")
def predict_year(year: int):
    predictions = []
    for country in countries:
        c = country_encoder.transform(np.array([country]))[0]
        X = pd.DataFrame.from_dict({"country_encoded": [c], "year": [year]}).astype("float32")
        prediction = model.predict(X)
        predictions.append({"country": country, "year": year, "TOE_HAB": float(prediction.values[0])})
    return predictions

@app.get("/health")
def health():
    return {"status": "ok"}