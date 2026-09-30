import numpy as np
import pandas as pd

TARGET_ROW = "Total International Visitor Arrivals By Place Of Residence"

def load_raw_data(csv_path):
    return pd.read_csv(csv_path)

def prepare_total_series(raw_df):
    """Extract the total tourist-arrivals series and convert wide data to long time-series data."""
    match = raw_df["DataSeries"].astype(str).str.strip().eq(TARGET_ROW)
    if not match.any():
        raise ValueError(f"Target series not found: {TARGET_ROW}")

    row = raw_df.loc[match].iloc[0]
    values = row.iloc[1:]
    dates = pd.to_datetime(values.index, format="%Y%b", errors="coerce")
    arrivals = pd.to_numeric(values.values, errors="coerce")

    df = pd.DataFrame({"Date": dates, "Tourist_Arrivals": arrivals})
    df = df.dropna(subset=["Date", "Tourist_Arrivals"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

def data_quality_report(raw_df, series_df):
    return {
        "raw_rows": int(raw_df.shape[0]),
        "raw_columns": int(raw_df.shape[1]),
        "raw_missing_cells": int(raw_df.isna().sum().sum()),
        "raw_duplicate_rows": int(raw_df.duplicated().sum()),
        "series_rows": int(series_df.shape[0]),
        "series_missing_values": int(series_df["Tourist_Arrivals"].isna().sum()),
        "series_duplicate_dates": int(series_df["Date"].duplicated().sum()),
        "start_date": series_df["Date"].min().strftime("%Y-%m"),
        "end_date": series_df["Date"].max().strftime("%Y-%m"),
    }

def add_features(df):
    out = df.copy()
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["Quarter"] = out["Date"].dt.quarter
    out["Month_Sin"] = np.sin(2 * np.pi * out["Month"] / 12)
    out["Month_Cos"] = np.cos(2 * np.pi * out["Month"] / 12)

    for lag in [1, 2, 3, 6, 12]:
        out[f"Lag_{lag}"] = out["Tourist_Arrivals"].shift(lag)

    for window in [3, 6, 12]:
        out[f"Rolling_Mean_{window}"] = (
            out["Tourist_Arrivals"].shift(1).rolling(window).mean()
        )
    return out

def make_model_data(series_df):
    featured = add_features(series_df).dropna().reset_index(drop=True)
    features = [
        "Year", "Month", "Quarter", "Month_Sin", "Month_Cos",
        "Lag_1", "Lag_2", "Lag_3", "Lag_6", "Lag_12",
        "Rolling_Mean_3", "Rolling_Mean_6", "Rolling_Mean_12"
    ]
    return featured, features

def make_future_features(history_df, next_date):
    """Create one recursive forecast row using actual + previously predicted history."""
    history = history_df.copy()
    history["Date"] = pd.to_datetime(history["Date"])
    history = history.sort_values("Date").reset_index(drop=True)

    month = next_date.month
    row = {
        "Year": next_date.year,
        "Month": month,
        "Quarter": next_date.quarter,
        "Month_Sin": np.sin(2 * np.pi * month / 12),
        "Month_Cos": np.cos(2 * np.pi * month / 12),
    }

    values = history["Tourist_Arrivals"]
    for lag in [1, 2, 3, 6, 12]:
        row[f"Lag_{lag}"] = float(values.iloc[-lag])

    for window in [3, 6, 12]:
        row[f"Rolling_Mean_{window}"] = float(values.iloc[-window:].mean())

    return pd.DataFrame([row])
