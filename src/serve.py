from fastapi import FastAPI
import pandas as pd 
import joblib 

MODEL_PATH = "models/alert_model.pkl"

FEATURES = [
    "pm2.5_lag_1",
    "pm2.5_lag_3",
    "pm2.5_lag_6",
    "temp",
    "pres",
    "wspm"
]

app= FastAPI(title="Beijing AQI Alert API",
             description="Predicts PM2.5-based air pollution alerts",
             version='1.0'
)

# Load model once at startup
model= joblib.load(MODEL_PATH)

@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_alert(payload: dict):
    """
    Expected JSON payload:
    {
      "pm2.5_lag_1": 160,
      "pm2.5_lag_3": 155,
      "pm2.5_lag_6": 148,
      "temp": -2.0,
      "pres": 1020,
      "wspm": 1.5
    }
    """
    df = pd.DataFrame([payload])
    pred = model.predict(df[FEATURES])[0]

    return {
        "alert": int(pred),
        "meaning": "High pollution alert" if pred == 1 else "Normal air quality"
    }
