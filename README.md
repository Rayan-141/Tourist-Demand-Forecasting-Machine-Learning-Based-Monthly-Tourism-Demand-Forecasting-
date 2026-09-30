# 🌏 Tourist Demand Forecasting Using Machine Learning

## Case Study 117

**Project title:** Tourist Demand Forecasting for Singapore Using Machine Learning

### Dataset
`International_Visit_Or_Arrivals_By_Place_Of_Residence_Monthly.csv`

Source: Singapore Department of Statistics (SINGSTAT)  
Original source: Singapore Tourism Board  
Official portal: https://data.gov.sg/datasets/d_7e7b2ee60c6ffc962f80fef129cf306e/view

## ML formulation
- Supervised learning
- Regression
- Time-series forecasting

## Models
1. Linear Regression
2. Polynomial Regression Degree 2
3. Polynomial Regression Degree 4
4. Random Forest Regression

## Preprocessing
- Wide-to-long transformation
- Date conversion
- Missing-value checks
- Duplicate checks
- Time features
- Lag features
- Rolling averages
- One-hot encoding for Month and Quarter
- Standard scaling for numeric features
- Chronological train/test split
- TimeSeriesSplit cross-validation

## Evaluation
- MAE
- MSE
- RMSE
- R²
- Time-series cross-validation
- Actual vs predicted analysis

## Run in VS Code

### 1. Create environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Run analysis
```bash
python3 analysis.py
```

### 4. Run Streamlit
```bash
streamlit run app.py
```

The browser will open the interactive application.

## No .pkl model files
The project intentionally does not save serialized model files. `app.py` trains the models from `models.py` and uses Streamlit's `st.cache_resource` to keep them in memory while the app is running.
