import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocessing import preprocess_timeseries
import math, warnings
warnings.filterwarnings("ignore")

def mean_absolute_percentage_error(y_true, y_pred):
    """Return MAPE in % (ignores any zero targets to avoid division error)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_holtwinter(filepath, trend="add", seasonal="add", seasonal_periods=252):
    # 1️⃣ Preprocess
    df, d, exog_features, scaler = preprocess_timeseries(filepath, apply_scaling=False)

    df["y_raw"] = df["y"]
    df['ds'] = df['ds'].dt.tz_localize(None)

    # 2️⃣ Train/Test Split
    split = int(len(df)*0.7)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    y_train, y_test = train_df["y"].values, test_df["y"].values

    # 3️⃣ Fit Holt-Winters
    model = ExponentialSmoothing(
        y_train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    ).fit()
    print(f"✅ Holt-Winters fitted ({trend}, {seasonal}) | differencing used: d={d}")

    # 4️⃣ Forecast
    y_pred = model.forecast(len(test_df))

    # 5️⃣ Accuracy
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("\n===== Holt-Winters Accuracy =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # 6️⃣ Plot
    plt.figure(figsize=(12,6))
    plt.plot(train_df['ds'], train_df['y_raw'], label="Train")
    plt.plot(test_df['ds'], y_test, label="Test", color="orange")
    plt.plot(test_df['ds'], y_pred, label="Holt-Winters Forecast", color="green")
    plt.title(f"Holt-Winters Forecast ({trend}, {seasonal})")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend(); plt.tight_layout(); plt.show()

    return y_pred, y_test, mae, rmse, mape

if __name__ == "__main__":
    filepath = "AAPL_1980-09-08_to_2025-09-11.csv"
    _, _, mae, rmse, mean_absolute_percentage_error = train_holtwinter(filepath)
    print(f"\nFinal → MAE: {mae:.4f}, RMSE: {rmse:.4f}")
