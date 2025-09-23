# compare_models.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
from preprocessing import preprocess_timeseries
import warnings

warnings.filterwarnings("ignore")

# ============================
# Evaluation Function
# ============================
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"Model": model_name, "MAE": mae, "RMSE": rmse}

# ============================
# Prophet
# ============================
def run_prophet(train_df, test_df, exog_features):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    for col in ['is_holiday', 'is_weekend']:
        if col in exog_features:
            model.add_regressor(col)

    model.fit(train_df[['ds', 'y'] + [c for c in exog_features if c in ['is_holiday', 'is_weekend']]])
    future = test_df[['ds'] + [c for c in exog_features if c in ['is_holiday', 'is_weekend']]].copy()
    forecast = model.predict(future)
    return forecast['yhat']

# ============================
# ARIMA
# ============================
def run_arima(train_series, test_series):
    model = auto_arima(
        train_series, 
        seasonal=False, 
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    preds = model.predict(n_periods=len(test_series))
    return preds

# ============================
# SARIMA
# ============================
def run_sarima(train_series, test_series, seasonal_period=5):
    model = auto_arima(
        train_series, 
        seasonal=True, 
        m=seasonal_period,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    preds = model.predict(n_periods=len(test_series))
    return preds

# ============================
# Holt-Winters
# ============================
def run_holtwinter(train_series, test_series):
    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add",
        seasonal_periods=252
    ).fit()
    preds = model.forecast(len(test_series))
    return preds

# ============================
# VAR
# ============================
def run_var(train_df, test_df):
    # Combine train and test for VAR dataset
    var_data = pd.concat([train_df, test_df])[["y", "lag_1", "lag_2", "lag_3"]].dropna()
    split_idx = len(train_df)
    train = var_data.iloc[:split_idx]
    test = var_data.iloc[split_idx:]

    model = VAR(train)
    results = model.fit(maxlags=5)
    forecast_input = train.values[-results.k_ar:]
    forecast_vals = results.forecast(forecast_input, steps=len(test))
    forecast_df = pd.DataFrame(forecast_vals, index=test.index, columns=var_data.columns)
    return forecast_df["y"].values

# ============================
# Main
# ============================
if __name__ == "__main__":
    filepath = "AAPL_1980-09-08_to_2025-09-11.csv"

    # Preprocess
    df, d, exog_features, scaler = preprocess_timeseries(filepath, apply_scaling=False)
    df["y_raw"] = df["y"].copy()
    df["y"] = df["y_raw"]
    df['ds'] = df['ds'].dt.tz_localize(None)

    # Train/test split (70:30)
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    y_actual = test_df["y_raw"].copy()

    results = []

    # Prophet
    y_pred_prophet = run_prophet(train_df, test_df, exog_features)
    results.append(evaluate(y_actual, y_pred_prophet, "Prophet"))

    # ARIMA
    y_pred_arima = run_arima(train_df['y'], test_df['y'])
    results.append(evaluate(y_actual, y_pred_arima, "ARIMA"))

    # SARIMA
    y_pred_sarima = run_sarima(train_df['y'], test_df['y'], seasonal_period=5)
    results.append(evaluate(y_actual, y_pred_sarima, "SARIMA"))

    # Holt-Winters
    y_pred_hw = run_holtwinter(train_df['y'], test_df['y'])
    results.append(evaluate(y_actual, y_pred_hw, "Holt-Winters"))

    # VAR
    y_pred_var = run_var(train_df, test_df)
    results.append(evaluate(y_actual, y_pred_var, "VAR"))

    # Summary Table
    results_df = pd.DataFrame(results)
    print("\n===== Model Comparison =====")
    print(results_df)

    # ============================
    # Line Plot
    # ============================
    plt.figure(figsize=(14, 7))
    plt.plot(train_df['ds'], train_df['y'], label="Train")
    plt.plot(test_df['ds'], y_actual, label="Test", color="black")

    plt.plot(test_df['ds'], y_pred_prophet, label="Prophet", alpha=0.7)
    plt.plot(test_df['ds'], y_pred_arima, label="ARIMA", alpha=0.7)
    plt.plot(test_df['ds'], y_pred_sarima, label="SARIMA", alpha=0.7)
    plt.plot(test_df['ds'], y_pred_hw, label="Holt-Winters", alpha=0.7)
    plt.plot(test_df['ds'], y_pred_var, label="VAR", alpha=0.7)

    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.title("Forecast Comparison Across Models")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================
    # Bar Charts: MAE & RMSE
    # ============================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(results_df["Model"], results_df["MAE"])
    axes[0].set_title("Model Comparison - MAE")
    axes[0].set_ylabel("MAE ($)")

    axes[1].bar(results_df["Model"], results_df["RMSE"])
    axes[1].set_title("Model Comparison - RMSE")
    axes[1].set_ylabel("RMSE ($)")

    plt.suptitle("Error Comparison Across 5 Models")
    plt.tight_layout()
    plt.show()
