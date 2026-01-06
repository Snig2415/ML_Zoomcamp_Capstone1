import joblib
import pandas as pd


MODEL_PATH = "models/alert_model.pkl"

FEATURES = [
    "pm2.5_lag_1",
    "pm2.5_lag_3",
    "pm2.5_lag_6",
    "temp",
    "pres",
    "wspm"
]


def load_model():
    return joblib.load(MODEL_PATH)


def predict(df):
    model = load_model()
    preds = model.predict(df[FEATURES])
    return preds


if __name__ == "__main__":
    # Example inference
    sample = pd.DataFrame([{
        "pm2.5_lag_1": 160,
        "pm2.5_lag_3": 155,
        "pm2.5_lag_6": 148,
        "temp": -2.0,
        "pres": 1020,
        "wspm": 1.5
    }])

    prediction = predict(sample)
    print("Alert prediction:", int(prediction[0]))
