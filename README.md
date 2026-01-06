
# 🌫️ Beijing AQI Alert Prediction


## ML Zoomcamp Capstone Project-1

End-to-end Machine Learning system for predicting hourly air-quality alerts using time-series data, deployed as a Dockerized FastAPI service.

## 📁 Project Structure:

```text
 AQI/
 ├── data/
 ├── notebooks/
 ├── src/
 ├── models/
 ├── Dockerfile
 └── README.md

```


## 1.1 🎯 Problem description 

Air pollution is a major public-health risk.
This project predicts whether PM2.5 concentration exceeds a hazardous threshold (≥150 μg/m³) for a given hour.

ML Task: Binary classification
Target variable:

alert = 1  if PM2.5 ≥ 150
alert = 0  otherwise

![alt text](image.png)

Evaluation metric: F1-score (recommended for imbalanced classification)

## 1.2 📊 Dataset

Source:
UCI Machine Learning Repository – Beijing Multi-Site Air Quality Dataset

Characteristics:

Hourly data (2013–2017)

12 monitoring stations

Pollutants + weather features

Raw data path: data/PRSA_Data_20130301-20170228/

## 1.3 🔍 Exploratory Data Analysis

EDA is performed in notebooks/01_eda.ipynb.

Key findings:

Strong temporal autocorrelation in PM2.5  

Severe class imbalance for pollution alerts  

Seasonal effects (winter pollution spikes)  

These findings directly motivated: 

Time-based splitting  

F1-score optimization  

## 1.4 🧠 Feature Engineering

Performed in notebooks/02_feature_engineering.ipynb.  

Leakage-safe design  

Lag features computed per station,No future data used for prediction  

Core features:  

pm2.5_lag_1  

pm2.5_lag_3  

pm2.5_lag_6  

Weather: temp, pres, wspm  

Processed datasets saved as CSV: data/processed/hourly_features.csv  
                                 data/processed/daily_features.csv  

## 1.5 🔐 Data Leakage :

Strict time-based splits are used:  

Train: < 2016-01-01  
Valid: 2016-01-01 → 2016-06-30  
Test:  ≥ 2016-07-01  

## 1.6 🏋️ Model Training:  

Model: Logistic Regression  
Why: Strong baseline, interpretable, handles imbalance well  

Training is done via script   
code: python src/train.py  
  
Outputs:  
  
Validation F1-score  
Trained model saved to: models/alert_model.pkl  
  
## 1.7 🔮 Inference  
  
Inference logic is isolated in: src/predict.py  

Loads trained model | Accepts structured input | Returns alert prediction  

Example input:  
json: {  
  "pm2.5_lag_1": 160,  
  "pm2.5_lag_3": 155,  
  "pm2.5_lag_6": 148,  
  "temp": -2.0,  
  "pres": 1020,  
  "wspm": 1.5
}  

## 1.8 🚀 API Service/Deployment  

Framework: FastAPI  
File: src/serve.py  
Run locally: uvicorn src.serve:app --reload  
Swagger UI:  http://localhost:8000/docs  

## 1.9 🐳 Docker (Containerization)  
  
Build image:   docker build -t beijing-aqi-alert .    
Run container: docker run -p 8000:8000 beijing-aqi-alert  
  
## 1.10 🧱 Architecture Diagram  

1) Training Architecture:

```text
┌──────────────┐
│ Raw CSV Data │
└──────┬───────┘
       │
       ▼
┌────────────────────────┐
│ Feature Engineering     │
│ (station-wise lags)     │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Time-based Split        │
│ Train / Val / Test      │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Logistic Regression     │
│ (F1 optimization)       │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ alert_model.pkl         │
└────────────────────────┘

```

2) Inference & Deployment Architecture

        Client / Swagger UI
                │
                ▼
        ┌─────────────────┐
        │ FastAPI Service  │
        │  (/predict)     │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ Trained ML Model │
        │ alert_model.pkl  │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ Prediction JSON │
        │   alert: 0 / 1  │
        └─────────────────┘






