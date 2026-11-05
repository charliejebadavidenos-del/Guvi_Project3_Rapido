from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Rapido Prediction API", version="1.0")

# Load models once at startup
clf = joblib.load('cancel_model.pkl')
reg = joblib.load('fare_model.pkl')
clf_cols = joblib.load('cancel_cols.pkl')
reg_cols = joblib.load('fare_cols.pkl')

class CancelInput(BaseModel):
    hour_of_day: int
    traffic_level: str
    weather_condition: str
    surge_multiplier: float
    ride_distance_km: float
    vehicle_type: str

class FareInput(BaseModel):
    ride_distance_km: float
    estimated_ride_time_min: int
    surge_multiplier: float
    hour_of_day: int
    traffic_level: str
    vehicle_type: str

@app.get("/")
def home():
    return {"message": "Rapido Prediction API is running"}

@app.post("/predict/cancellation")
def predict_cancellation(data: CancelInput):
    df = pd.DataFrame([data.dict()])
    df_encoded = pd.get_dummies(df).reindex(columns=clf_cols, fill_value=0)
    prob = clf.predict_proba(df_encoded)[0][1]
    return {
        "cancellation_probability": round(prob * 100, 2),
        "risk_level": "High" if prob > 0.4 else "Low"
    }

@app.post("/predict/fare")
def predict_fare(data: FareInput):
    df = pd.DataFrame([data.dict()])
    df_encoded = pd.get_dummies(df).reindex(columns=reg_cols, fill_value=0)
    fare = reg.predict(df_encoded)[0]
    return {
        "predicted_fare": round(float(fare), 2),
        "currency": "INR"
    }