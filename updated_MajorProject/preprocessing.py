import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from io import StringIO
from scipy.stats import skew

# ------------------------------------------------------------------
# 1️⃣  Robust CSV Reader
# ------------------------------------------------------------------
def smart_read_csv(path: str) -> pd.DataFrame:
    """
    Reads a CSV even when:
      • The real header is not on the first line.
      • There are extra rows like tickers or comments.
    Scans the first few lines to find a header with at least 2 columns of data.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Try different header rows until we find a plausible one
    for i in range(min(10, len(lines))):
        try:
            trial = pd.read_csv(StringIO("".join(lines)), header=i)
            if trial.shape[1] > 1:  # at least 2 columns
                return trial
        except Exception:
            continue
    # Fallback: default read
    return pd.read_csv(path)

# ------------------------------------------------------------------
# 2️⃣  Flexible Date Detection
# ------------------------------------------------------------------
def detect_date_column(df: pd.DataFrame) -> str:
    """
    Detects a date/time column by:
      1. Column name keywords.
      2. Value inspection (parse-ability as datetime).
      3. If none found, creates a synthetic business-day timeline.
    Returns the detected or created column name.
    """
    # 1. Name-based quick detection
    for c in df.columns:
        if any(k in c.lower() for k in ["date", "time", "ds", "timestamp"]):
            return c

    # 2. Value-based detection: look for columns whose values mostly parse as datetime
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(50)
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() > 0.8:  # 80% look like dates
            return c

    # 3. No date-like column found: create synthetic dates
    print("⚠️ No date-like column found. Creating synthetic business-day index.")
    df.insert(0, "SyntheticDate",
              pd.date_range(start=pd.Timestamp.today().normalize() - pd.offsets.BDay(len(df)-1),
                            periods=len(df), freq="B"))
    return "SyntheticDate"

# ------------------------------------------------------------------
# 3️⃣  Main Preprocessing Function with EMA Smoothing
# ------------------------------------------------------------------
def preprocess_timeseries(filepath: str,
                          apply_scaling: bool = False,
                          max_diff: int = 2,
                          #apply_noise_reduction: bool = True,
                          ema_span: int = 20):
    """
    Preprocess stock time series data:
      • Robust CSV reading (handles extra header/ticker rows)
      • Auto date column detection by name or values, with synthetic fallback
      • Converts numeric columns
      • Reports row counts, missing data, duplicates, skewness, and stationarity metrics
      • Resamples to business-day frequency and interpolates
      • Adds lag and calendar features
      • Stationarity differencing with updated ADF p-values
      • Optional Exponential Moving Average (EMA) smoothing for noise reduction
      • Optional StandardScaler for exogenous features

    Returns:
        df  : cleaned DataFrame with 'ds' (datetime), 'y' (target), and optionally 'y_smooth' (EMA-smoothed target)
        d   : number of differences applied
        exog_features : list of exogenous feature names
        scaler : fitted StandardScaler or None
    """

    # ---------- 1. Load CSV intelligently ----------
    df = smart_read_csv(filepath)
    print(f"📊 Initial rows in raw CSV (including header row guess): {len(df)}")

    # ---------- 2. Detect and standardize date column ----------
    date_col = detect_date_column(df)
    df.rename(columns={date_col: "ds"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    # ---------- 3. Identify target column ----------
    possible_targets = [c for c in df.columns
                        if c.lower() in ["close", "adj_close", "price"]]
    if possible_targets:
        target_col = possible_targets[0]
    else:
        # Fallback: first numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns
        if not len(num_cols):
            raise ValueError("❌ No numeric target column found.")
        target_col = num_cols[0]

    df.rename(columns={target_col: "y"}, inplace=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")   # ensure numeric

    # ---------- 4. Report missing and duplicates BEFORE cleaning ----------
    missing_before = df.isna().sum().sum()
    dup_before = df.duplicated().sum()
    print(f"🔍 Missing values before cleaning: {missing_before}")
    print(f"🔍 Duplicate rows before cleaning: {dup_before}")

    # Drop NA in ds to focus on valid time rows
    # Any non-numeric entries in the target column y become NaN.
    # Rows without a valid datetime (ds) are removed.
    df.dropna(subset=["ds"], inplace=True)

    # Remove duplicates on full rows
    df_before_drop = len(df)
    df.drop_duplicates(inplace=True)
    dup_removed = df_before_drop - len(df)
    if dup_removed:
        print(f"🗑️ Removed {dup_removed} duplicate rows. Remaining rows: {len(df)}")
    else:
        print("✅ No duplicate rows found.")

    # ---------- 5. Sort and interpolate ----------
    df.sort_values("ds", inplace=True)

    # ---- Outlier Clipping using IQR ----
    q1, q3 = df["y"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    before_clip = len(df)
    df["y"] = np.clip(df["y"], lower, upper)
    print(f"✂️ Outlier clipping applied using IQR. Bounds: [{lower:.2f}, {upper:.2f}]")

    # ---------- 6. Skewness before differencing ----------
    skew_before = df["y"].dropna().skew()
    print(f"📈 Skewness before preprocessing: {skew_before:.4f}")

    # ---------- 7. Resample to Business Days(Business-Day Resampling + Interpolation) ----------
    df.set_index("ds", inplace=True)
    df = df.asfreq("B")  # inserts missing business-day rows
    df["y"] = df["y"].interpolate(method="time") # time-based linear interpolation
    df.reset_index(inplace=True)

    # ---- Automatic Noise Detection ----
    # Compute short-term vs long-term volatility ratio
    short_vol = df["y"].rolling(window=5).std().mean()
    long_vol = df["y"].rolling(window=20).std().mean()
    noise_ratio = short_vol / long_vol if long_vol and not np.isnan(long_vol) else 0
    if noise_ratio > 1.2: # heuristic threshold
     df["y_smooth"] = df["y"].ewm(span=ema_span, adjust=False).mean()
     print(f"✨ High noise detected (ratio={noise_ratio:.2f}). Applied EMA smoothing span={ema_span}.")
    else:
     print(f"✅ Noise level acceptable (ratio={noise_ratio:.2f}). No smoothing applied.")
    
    # ---------- 9. Feature Engineering ----------
    #1.Lags
    for lag in range(1, 4):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    # 2. Rolling statistics (mean & std over 7 and 30 day windows)
    df["roll_mean_7"] = df["y"].rolling(window=7).mean()
    df["roll_std_7"] = df["y"].rolling(window=7).std()
    df["roll_mean_30"] = df["y"].rolling(window=30).mean()
    df["roll_std_30"] = df["y"].rolling(window=30).std()
    #3.Seasonal signals
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year
    df.dropna(inplace=True)

    # ---------- 10. Stationarity Check & Differencing ----------
    adf_stat, pval, *_ = adfuller(df["y"].dropna())
    print(f"📊 ADF Statistic before differencing: {adf_stat:.4f}, p-value: {pval:.10f}")

    d = 0
    while pval > 0.05 and d < max_diff:
        df["y"] = df["y"].diff().bfill()
        d += 1
        adf_stat, pval, *_ = adfuller(df["y"].dropna())
        print(f"   ↳ After differencing {d}: ADF {adf_stat:.4f}, p-value {pval:.10f}")

    if pval > 0.05:
        print("⚠️ Max differencing reached; data may remain non-stationary.")
    else:
        print(f"✅ Stationarity achieved with d={d}")

    # ---------- 11. Skewness after differencing ----------
    skew_after = df["y"].dropna().skew()
    print(f"📉 Skewness after preprocessing: {skew_after:.4f}")

    # ---------- 12. Optional Scaling ----------
    exog_features = [c for c in df.columns if c not in ["ds", "y"]]
    scaler = None
    if apply_scaling and exog_features:
        scaler = StandardScaler()
        df[exog_features] = scaler.fit_transform(df[exog_features])
        print("✅ Scaling applied to exogenous features")
    else:
        print("ℹ️ Scaling skipped (fine for ARIMA/SARIMA/Prophet/Holt-Winters)")

    # Final row count after all cleaning
    print(f"📊 Final number of rows after preprocessing: {len(df)}")

    return df, d, exog_features, scaler

# ------------------------------------------------------------------
# Quick test when run directly
# ------------------------------------------------------------------
# if __name__ == "__main__":
#     path = "AAPL_1980-09-08_to_2025-09-11.csv"  # or any of your CSVs
#     df, d, exog_features, scaler = preprocess_timeseries(path, apply_scaling=False)
#     print(df.head())
#     print(f"\nDifferencing order used: d={d}")
#     print(f"Exogenous features: {exog_features}")
