from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")   # pipeline model


class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


@app.get("/")
def home():
    return {"status": "Running", "model": "House Price Predictor"}


@app.post("/predict")
def predict(features: HouseFeatures):

    data = pd.DataFrame([{
        "longitude": features.longitude,
        "latitude": features.latitude,
        "housing_median_age": features.housing_median_age,
        "total_rooms": features.total_rooms,
        "total_bedrooms": features.total_bedrooms,
        "population": features.population,
        "households": features.households,
        "median_income": features.median_income,
        "ocean_proximity": features.ocean_proximity
    }])

    prediction = model.predict(data)[0]

    return {"predicted_price": float(prediction)}
