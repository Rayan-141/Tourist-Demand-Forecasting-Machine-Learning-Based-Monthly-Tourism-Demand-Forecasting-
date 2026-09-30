import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from data_utils import add_features, make_model_data, make_future_features

FEATURES = [
    "Year", "Month", "Quarter", "Month_Sin", "Month_Cos",
    "Lag_1", "Lag_2", "Lag_3", "Lag_6", "Lag_12",
    "Rolling_Mean_3", "Rolling_Mean_6", "Rolling_Mean_12"
]
CATEGORICAL = ["Month", "Quarter"]
NUMERIC = [x for x in FEATURES if x not in CATEGORICAL]

def build_linear_model():
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

def build_random_forest():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

def build_polynomial_model(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

def chronological_split(featured_df, test_size=0.20):
    split = int(len(featured_df) * (1 - test_size))
    return featured_df.iloc[:split].copy(), featured_df.iloc[split:].copy()

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred)
    }

def train_and_compare(featured_df):
    train_df, test_df = chronological_split(featured_df)
    X_train, y_train = train_df[FEATURES], train_df["Tourist_Arrivals"]
    X_test, y_test = test_df[FEATURES], test_df["Tourist_Arrivals"]

    models = {
        "Linear Regression": build_linear_model(),
        "Random Forest": build_random_forest(),
    }

    rows = []
    fitted = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = evaluate(y_test, pred)
        rows.append({"Model": name, **metrics})
        fitted[name] = model
        predictions[name] = pred

    # Polynomial regression is deliberately a simple time-trend experiment,
    # matching the syllabus's degree-2 vs degree-4 curve-fitting activity.
    time_train = np.arange(len(train_df)).reshape(-1, 1)
    time_test = np.arange(len(train_df), len(featured_df)).reshape(-1, 1)

    for degree in [2, 4]:
        model = build_polynomial_model(degree)
        model.fit(time_train, y_train)
        pred = model.predict(time_test)
        metrics = evaluate(y_test, pred)
        rows.append({"Model": f"Polynomial Regression (Degree {degree})", **metrics})
        fitted[f"Polynomial Regression (Degree {degree})"] = model
        predictions[f"Polynomial Regression (Degree {degree})"] = pred

    results = pd.DataFrame(rows)
    actual_vs_pred = test_df[["Date", "Tourist_Arrivals"]].copy()
    for name, pred in predictions.items():
        actual_vs_pred[name] = pred

    return results, fitted, actual_vs_pred, train_df, test_df

def time_series_cv(featured_df, n_splits=5):
    X = featured_df[FEATURES]
    y = featured_df["Tourist_Arrivals"]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []

    for model_name in ["Linear Regression", "Random Forest"]:
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            model = build_linear_model() if model_name == "Linear Regression" else build_random_forest()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            m = evaluate(y.iloc[test_idx], pred)
            records.append({"Model": model_name, "Fold": fold, **m})

    return pd.DataFrame(records)

def random_forest_feature_importance(model, feature_names=FEATURES):
    """Return Random Forest feature importance values."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError("The supplied model is not a fitted Random Forest.")
    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

def fit_final_models(featured_df):
    X = featured_df[FEATURES]
    y = featured_df["Tourist_Arrivals"]
    models = {
        "Linear Regression": build_linear_model(),
        "Random Forest": build_random_forest(),
    }
    for model in models.values():
        model.fit(X, y)
    return models

def recursive_forecast(model, history_df, months=12):
    history = history_df[["Date", "Tourist_Arrivals"]].copy()
    history["Date"] = pd.to_datetime(history["Date"])
    history = history.sort_values("Date").reset_index(drop=True)

    forecasts = []
    last_date = history["Date"].iloc[-1]

    for _ in range(months):
        next_date = last_date + pd.DateOffset(months=1)
        X_next = make_future_features(history, next_date)
        prediction = float(model.predict(X_next[FEATURES])[0])
        prediction = max(0.0, prediction)

        forecasts.append({
            "Date": next_date,
            "Predicted_Tourist_Arrivals": prediction
        })

        history = pd.concat([
            history,
            pd.DataFrame({"Date": [next_date], "Tourist_Arrivals": [prediction]})
        ], ignore_index=True)
        last_date = next_date

    return pd.DataFrame(forecasts)
