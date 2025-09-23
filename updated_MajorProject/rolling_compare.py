# rolling_cv_compare.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from pmdarima import auto_arima
from preprocessing import preprocess_timeseries

def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def run_prophet(train_df, test_df, exog_features):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    for col in ['is_holiday', 'is_weekend']:
        if col in exog_features:
            model.add_regressor(col)

    model.fit(train_df[['ds', 'y'] + [c for c in exog_features if c in ['is_holiday', 'is_weekend']]])
    future = test_df[['ds'] + [c for c in exog_features if c in ['is_holiday', 'is_weekend']]].copy()
    forecast = model.predict(future)
    return forecast['yhat']


def run_arima(train_series, test_series):
    model = auto_arima(
        train_series, 
        seasonal=False, 
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    return model.predict(n_periods=len(test_series))


def run_sarima(train_series, test_series, seasonal_period=5):
    model = auto_arima(
        train_series, 
        seasonal=True, 
        m=seasonal_period,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    return model.predict(n_periods=len(test_series))


if __name__ == "__main__":
    filepath = "AAPL_1980-12-03_2025-04-06.csv"

    # Preprocess
    df, d, exog_features, scaler = preprocess_timeseries(filepath, apply_scaling=False)
    df["y_raw"] = df["y"].copy()
    df["y"] = df["y_raw"]
    df['ds'] = df['ds'].dt.tz_localize(None)

    # ============================
    # Rolling Window CV
    # ============================
    n_splits = 5   # number of folds
    test_size = int(len(df) * 0.1)  # 10% each fold
    step_size = test_size

    results = []

    for fold in range(n_splits):
        train_end = len(df) - (n_splits - fold) * test_size
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:train_end+test_size].copy()
        y_actual = test_df["y_raw"].copy()

        # Prophet
        y_pred_prophet = run_prophet(train_df, test_df, exog_features)
        mae, rmse = evaluate(y_actual, y_pred_prophet, "Prophet")
        results.append({"Fold": fold+1, "Model": "Prophet", "MAE": mae, "RMSE": rmse})

        # ARIMA
        y_pred_arima = run_arima(train_df['y'], test_df['y'])
        mae, rmse = evaluate(y_actual, y_pred_arima, "ARIMA")
        results.append({"Fold": fold+1, "Model": "ARIMA", "MAE": mae, "RMSE": rmse})

        # SARIMA
        y_pred_sarima = run_sarima(train_df['y'], test_df['y'], seasonal_period=5)
        mae, rmse = evaluate(y_actual, y_pred_sarima, "SARIMA")
        results.append({"Fold": fold+1, "Model": "SARIMA", "MAE": mae, "RMSE": rmse})

        print(f"✅ Fold {fold+1} complete")

    # Results DataFrame
    results_df = pd.DataFrame(results)
    print("\n===== Rolling Forecast CV Results =====")
    print(results_df)

    # Average scores
    avg_results = results_df.groupby("Model")[["MAE","RMSE"]].mean().reset_index()
    print("\n===== Average CV Performance =====")
    print(avg_results)

    # Plot results
    plt.figure(figsize=(10,6))
    for model in avg_results["Model"]:
        plt.plot(results_df[results_df["Model"]==model]["Fold"], 
                 results_df[results_df["Model"]==model]["RMSE"], 
                 marker="o", label=model)
    plt.xlabel("Fold")
    plt.ylabel("RMSE ($)")
    plt.title("Rolling Forecast CV - RMSE per Fold")
    plt.legend()
    plt.tight_layout()
    plt.show()
