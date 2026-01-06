import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


DATA_PATH = "data/processed/hourly_features.csv"
MODEL_PATH = "models/alert_model.pkl"


def load_data(path):
    df = pd.read_csv(
        path,
        parse_dates=["datetime"],
        index_col="datetime"
    )
    return df


def split_data(df):
    train = df[df.index < "2016-01-01"]
    val   = df[(df.index >= "2016-01-01") &
               (df.index < "2016-07-01")]
    test  = df[df.index >= "2016-07-01"]
    return train, val, test


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    features = [
        "pm2.5_lag_1",
        "pm2.5_lag_3",
        "pm2.5_lag_6",
        "temp",
        "pres",
        "wspm"
    ]

    train, val, test = split_data(df)

    X_train = train[features]
    y_train = train["alert"]

    X_val = val[features]
    y_val = val["alert"]

    print("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    f1 = f1_score(y_val, val_preds)

    print(f"Validation F1: {f1:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
