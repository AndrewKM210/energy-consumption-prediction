from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import os


API_SECRET = os.environ.get("ENERGY_API_SECRET")


# Load model and country encoder
model = joblib.load("trained_models/xgb_model.pkl")
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
    return {"country": country, "year": year, "TOE_HAB": float(model.predict(X))}


# Returns the predicted TOE_HAB consumption of all EU countries for a specified year
@app.get("/predict-year")
def predict_year(year: int, authorization: str = Header(None)):
    if authorization != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    predictions = []
    for country in countries:
        c = country_encoder.transform(np.array([country]))[0]
        X = pd.DataFrame.from_dict({"country_encoded": [c], "year": [year]}).astype("float32")
        predictions.append({"country": country, "year": year, "TOE_HAB": float(model.predict(X))})
    return predictions


@app.get("/health")
def health(authorization: str = Header(None)):
    if authorization != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "ok"}
