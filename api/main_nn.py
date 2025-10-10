from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from ml_training.nn_model import NNModel  # noqa: F401
import os


API_SECRET = os.environ.get("ENERGY_API_SECRET")


# Load model and country encoder
model = joblib.load("trained_models/neural_network.pkl")
countries = joblib.load("trained_models/countries.pkl")
country_encoder = joblib.load("trained_models/country_encoder.pkl")

# Initialize API
app = FastAPI(title="Energy Forecast API")

# TODO: posibly limit to only the streamlit dashboard
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Returns the predicted TOE_HAB consumption of a country for a specified year
@app.get("/predict")
def predict(country: str, year: int, authorization: str = Header(None)):
    if authorization != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    c = country_encoder.transform(np.array([country]))[0]
    X = pd.DataFrame.from_dict({"country_encoded": [c], "year": [year]}).astype("float32")
    prediction = model.predict(X)
    return {"country": country, "year": year, "TOE_HAB": float(prediction.values[0])}


# Returns the predicted TOE_HAB consumption of all EU countries for a specified year
@app.get("/predict-year")
def predict_year(year: int, authorization: str = Header(None)):
    if authorization != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    predictions = []
    for country in countries:
        c = country_encoder.transform(np.array([country]))[0]
        X = pd.DataFrame.from_dict({"country_encoded": [c], "year": [year]}).astype("float32")
        prediction = model.predict(X)
        predictions.append({"country": country, "year": year, "TOE_HAB": float(prediction.values[0])})
    return predictions

@app.get("/predict-country")
def predict_country(country: str, authorization: str = Header(None)):
    if authorization != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    y = list(range(2025,2035))
    c = [country_encoder.transform(np.array([country]))[0]]*len(y)
    df = pd.DataFrame({"country_encoded": c, "year": y})
    preds = model.predict(df).tolist()
    ret = [{"country": country, "year": y[i], "TOE_HAB": float(preds[i])} for i in range(len(preds))]
    return ret

@app.get("/health")
def health(authorization: str = Header(None)):
    if authorization != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "ok"}
