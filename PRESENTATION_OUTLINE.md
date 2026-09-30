# Presentation Outline — Tourist Demand Forecasting

## Slide 1 — Title
Tourist Demand Forecasting for Singapore Using Machine Learning

## Slide 2 — Problem Statement
A tourism organization wants to estimate demand for a destination/service over time.

## Slide 3 — Real-world justification
Forecasting can support tourism planning, capacity planning, staffing, transport and services.

## Slide 4 — Objectives
- Define the ML problem
- Select and document a real dataset
- Clean and preprocess data
- Perform EDA
- Build and compare ML models
- Evaluate errors
- Build Streamlit application

## Slide 5 — Dataset
International Visitor Arrivals By Place Of Residence, Monthly
SINGSTAT / Singapore Tourism Board

## Slide 6 — ML formulation
Supervised Learning → Regression → Time-Series Forecasting

## Slide 7 — Data preprocessing
Wide-to-long, dates, missing values, duplicates, encoding, scaling.

## Slide 8 — EDA
Trend, yearly totals, monthly seasonality, rolling averages, COVID disruption.

## Slide 9 — Feature engineering
Month, quarter, sine/cosine month, lag 1/2/3/6/12, rolling mean 3/6/12.

## Slide 10 — Models
Linear Regression
Polynomial Regression degree 2 and 4
Random Forest Regression

## Slide 11 — Evaluation
MAE, MSE, RMSE, R², chronological test set and TimeSeriesSplit.

## Slide 12 — Results
Show the actual generated metrics table from the notebook/app.

## Slide 13 — Forecast
Show the Streamlit future-demand forecast.

## Slide 14 — Limitations
External shocks, economic conditions, travel restrictions, data limitations.

## Slide 15 — Conclusion
End-to-end ML workflow from real tourism data to deployed Streamlit forecasting application.
