import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_utils import load_raw_data, prepare_total_series, data_quality_report, make_model_data
from models import train_and_compare, time_series_cv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "International_Visit_Or_Arrivals_By_Place_Of_Residence_Monthly.csv")

def main():
    raw = load_raw_data(CSV_PATH)
    series = prepare_total_series(raw)
    quality = data_quality_report(raw, series)

    print("DATA QUALITY REPORT")
    for k, v in quality.items():
        print(f"{k}: {v}")

    featured, _ = make_model_data(series)
    results, _, actual_vs_pred, train_df, test_df = train_and_compare(featured)
    print("\nMODEL COMPARISON")
    print(results.round(2).to_string(index=False))

    cv = time_series_cv(featured)
    print("\nTIME-SERIES CROSS-VALIDATION")
    print(cv.round(2).to_string(index=False))

    print("\nTEST SET PREDICTIONS")
    print(actual_vs_pred.tail(10).round(2).to_string(index=False))

    print("\nData saved/available through:", series["Date"].max().strftime("%Y-%m"))

if __name__ == "__main__":
    main()
