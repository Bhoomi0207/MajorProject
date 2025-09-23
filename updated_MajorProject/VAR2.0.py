import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocessing import preprocess_timeseries
import math, warnings
warnings.filterwarnings("ignore")

def mean_absolute_percentage_error(y_true, y_pred):
    """Return MAPE in % (ignores any zero targets to avoid division error)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_var(filepath):
    # Preprocess
    df, d, exog_features, scaler = preprocess_timeseries(filepath, apply_scaling=False)
    df['ds'] = df['ds'].dt.tz_localize(None)

    # Select multivariate columns: target + a couple of lag features
    cols = ['y', 'lag_1', 'lag_2', 'lag_3']
    df = df[['ds'] + [c for c in cols if c in df.columns]]

    # Drop any rows with NA after shifting
    df.dropna(inplace=True)

    split = int(len(df)*0.7)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    model = VAR(train_df.drop(columns=['ds']))
    fit = model.fit(maxlags=3)
    print(f"✅ VAR fitted | lags used: {fit.k_ar} | preprocessing d={d}")

    # Forecast
    lag_order = fit.k_ar
    input_data = train_df.drop(columns=['ds']).values[-lag_order:]
    forecast_vals = fit.forecast(y=input_data, steps=len(test_df))

    # We evaluate only the main 'y' column
    y_pred = forecast_vals[:,0]
    y_test = test_df['y'].values

    # Accuracy
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("\n===== VAR Accuracy =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(train_df['ds'], train_df['y'], label="Train")
    plt.plot(test_df['ds'], y_test, label="Test", color="orange")
    plt.plot(test_df['ds'], y_pred, label="VAR Forecast", color="green")
    plt.title("VAR Forecast")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend(); plt.tight_layout(); plt.show()

    return y_pred, y_test, mae, rmse, mape

if __name__ == "__main__":
    filepath = "AAPL_1980-09-08_to_2025-09-11.csv"
    _, _, mae, rmse, mean_absolute_percentage_error = train_var(filepath)
    print(f"Final → MAE: {mae:.4f}, RMSE: {rmse:.4f}")
