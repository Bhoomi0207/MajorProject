import pandas as pd
import exchange_calendars as ec

def check_date_gaps(filepath, date_col=None):
    # Load CSV
    df = pd.read_csv(filepath)

   # Auto-detect date column if not provided
    if date_col is None:
    # Look for any column name containing 'date' or 'time' (case-insensitive)
        possible_date_cols = [c for c in df.columns if any(x in c.lower() for x in ["date", "ds", "time"])]
        if not possible_date_cols:
            raise ValueError(f"❌ No date column found. Columns available: {list(df.columns)}")
        date_col = possible_date_cols[0]

    print(f"📌 Using '{date_col}' as the date column")

    # Convert to datetime with UTC then strip tz
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df.dropna(subset=[date_col], inplace=True)
    df.sort_values(by=date_col, inplace=True)

    # Strip timezone and normalize to midnight
    df[date_col] = df[date_col].dt.tz_localize(None)
    df[date_col] = df[date_col].dt.normalize()

    # BSE calendar
    bse_cal = ec.get_calendar("XBOM")

    # Clamp start_date to the first valid session of the calendar
    start_date = max(df[date_col].iloc[0], bse_cal.first_session)
    end_date = min(df[date_col].iloc[-1], bse_cal.last_session)

    expected_days = bse_cal.sessions_in_range(start_date, end_date)

    # Actual days
    actual_days = df[date_col].unique()

    # Missing days
    missing_days = expected_days.difference(actual_days)

    # Report
    print("\n===== Date Gap Report =====")
    print(f"📅 Date range: {df[date_col].min().date()} → {df[date_col].max().date()}")
    print(f"📊 Total rows in file: {len(df)}")
    print(f"🔁 Duplicate dates: {df.duplicated(subset=[date_col]).sum()}")
    print(f"📉 Expected BSE trading days: {len(expected_days)}")
    print(f"⚠️ Missing BSE trading days: {len(missing_days)}")

    if len(missing_days) > 0:
        print("\n❌ Missing Days (first 20 shown):")
        print(missing_days[:20])
    else:
        print("\n✅ No gaps found (matches BSE/NSE trading calendar)")

    print("\n===========================")


if __name__ == "__main__":
    filepath = "AAPL_2018-09-08_to_2025-09-08.csv"
    check_date_gaps(filepath)
