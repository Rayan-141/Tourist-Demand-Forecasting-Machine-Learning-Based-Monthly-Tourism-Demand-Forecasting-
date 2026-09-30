# Final Report Draft — Tourist Demand Forecasting for Singapore

## 1. Problem Definition
A tourism organization wants to estimate demand for a destination or service over time. This project formulates the problem as supervised regression/time-series forecasting of Singapore's monthly international visitor arrivals.

## 2. Dataset
Dataset: International Visitor Arrivals By Place Of Residence, Monthly.
Agency: Singapore Department of Statistics (SINGSTAT).
Original source: Singapore Tourism Board.
File used: International_Visit_Or_Arrivals_By_Place_Of_Residence_Monthly.csv.

The supplied CSV contains 62 data-series rows and 218 columns. The selected total series contains 217 monthly observations from 2008-01 through 2026-01.

## 3. Preprocessing
The original dataset is wide: months are columns. The total series was extracted and converted to a long format with Date and Tourist_Arrivals columns. Dates were converted to datetime, values were converted to numeric, duplicate dates were checked, and the data were sorted chronologically.

Feature engineering created Year, Month, Quarter, sine/cosine month features, Lag 1/2/3/6/12 and Rolling Mean 3/6/12.

Month and Quarter are one-hot encoded for the linear model. Numeric features are standardized. Random Forest is trained without requiring feature scaling.

## 4. EDA
The project examines the overall monthly trend, yearly totals, monthly seasonality and rolling averages. A major disruption is visible around 2020–2021, followed by recovery.

## 5. Models
- Linear Regression
- Polynomial Regression degree 2
- Polynomial Regression degree 4
- Random Forest Regression

Polynomial regression is included as a syllabus-aligned curve-fitting experiment. Random Forest is included as the ensemble model.

## 6. Evaluation
Because the target is numeric, regression metrics are used: MAE, MSE, RMSE and R². The final test set is the latest 20% of the chronological data. TimeSeriesSplit is used as a robustness check.

### Actual test-set results
```text
                           Model        MAE          MSE       RMSE      R2
               Linear Regression   94563.52 1.703357e+10  130512.71    0.65
                   Random Forest  184914.52 4.837165e+10  219935.55    0.02
Polynomial Regression (Degree 2) 2004034.26 4.424008e+12 2103332.71  -88.89
Polynomial Regression (Degree 4) 2368315.67 6.140220e+12 2477946.67 -123.76
```

### Time-series cross-validation summary
```text
                         MAE                  RMSE               R2      
                        mean        std       mean        std  mean   std
Model                                                                    
Linear Regression   97529.52   38919.97  150124.69   79203.91  0.45  0.39
Random Forest      184735.45  116872.80  257117.37  162530.30 -0.25  0.50
```

## 7. Error Analysis
Errors are inspected using actual-vs-predicted plots and the metric table. Large errors can occur around abrupt structural changes such as the COVID-era tourism disruption and recovery.

## 8. Streamlit Application
The Streamlit app provides Home, EDA, Model Comparison, Forecast and Project/Viva pages. Users can select a forecasting model and a 1–12 month horizon.

No .pkl file is used. Models are trained by Python code in models.py and cached in memory by Streamlit during the application session.

## 9. Limitations
The model uses historical visitor-arrival patterns and does not directly include all external drivers such as economic conditions, travel restrictions, airline capacity, weather, exchange rates or unexpected global events. Forecasts are therefore estimates rather than guarantees.

## 10. Conclusion
The project implements the complete ML workflow required by Case Study 117: problem definition, real dataset documentation, data cleaning, EDA, preprocessing, feature engineering, multiple regression models, evaluation, error analysis and a working Streamlit application.
