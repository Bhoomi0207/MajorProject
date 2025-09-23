import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocessing import preprocess_timeseries
import math, warnings
warnings.filterwarnings("ignore")

def mean_absolute_percentage_error(y_true, y_pred):
    """Return MAPE in % (ignores any zero targets to avoid division error)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_prophet(filepath):
    # Preprocess
    df, d, exog_features, scaler = preprocess_timeseries(filepath, apply_scaling=False)
    df = df[['ds', 'y']]        # Prophet only needs these columns
    df['ds'] = df['ds'].dt.tz_localize(None)

    # Train/Test Split
    split = int(len(df)*0.7)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    # Fit Prophet
    m = Prophet()
    m.fit(train_df)
    print(f"✅ Prophet model fitted | preprocessing d={d}")

    # Forecast
    future = test_df[['ds']]
    forecast = m.predict(future)
    y_pred = forecast['yhat'].values
    y_test = test_df['y'].values

    # Accuracy
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("\n===== Prophet Accuracy =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(train_df['ds'], train_df['y'], label="Train")
    plt.plot(test_df['ds'], y_test, label="Test", color="orange")
    plt.plot(test_df['ds'], y_pred, label="Prophet Forecast", color="green")
    plt.title("Prophet Forecast")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend(); plt.tight_layout(); plt.show()

    return y_pred, y_test, mae, rmse, mape

if __name__ == "__main__":
    filepath = "AAPL_1980-09-08_to_2025-09-11.csv"
    _, _, mae, rmse, mean_absolute_percentage_error = train_prophet(filepath)
    print(f"Final → MAE: {mae:.4f}, RMSE: {rmse:.4f}")
