from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="Rapido Prediction API", version="1.0")

# Load models once at startup
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

clf = joblib.load(MODEL_DIR / "cancel_model.pkl")
reg = joblib.load(MODEL_DIR / "fare_model.pkl")
clf_cols = joblib.load(MODEL_DIR / "cancel_cols.pkl")
reg_cols = joblib.load(MODEL_DIR / "fare_cols.pkl")

class CancelInput(BaseModel):
    hour_of_day: int
    traffic_level: str
    weather_condition: str
    surge_multiplier: float
    ride_distance_km: float
    vehicle_type: str
    city: str
    day_of_week: str
    is_weekend: int
    preferred_vehicle_type: str
    driver_experience_years: float

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