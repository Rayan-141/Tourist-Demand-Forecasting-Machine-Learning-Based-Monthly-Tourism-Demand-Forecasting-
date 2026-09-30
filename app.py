import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Make local modules importable when Streamlit runs from the project root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils import load_raw_data, prepare_total_series, data_quality_report, make_model_data
from models import train_and_compare, time_series_cv, fit_final_models, recursive_forecast, random_forest_feature_importance

st.set_page_config(
    page_title="Singapore Tourist Demand Forecasting",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(
    BASE_DIR, "data",
    "International_Visit_Or_Arrivals_By_Place_Of_Residence_Monthly.csv"
)

@st.cache_data
def load_project_data():
    raw = load_raw_data(CSV_PATH)
    series = prepare_total_series(raw)
    quality = data_quality_report(raw, series)
    featured, _ = make_model_data(series)
    return raw, series, featured, quality

@st.cache_resource
def train_project_models(featured):
    results, _, actual_vs_pred, train_df, test_df = train_and_compare(featured)
    final_models = fit_final_models(featured)
    cv = time_series_cv(featured)
    return results, actual_vs_pred, train_df, test_df, final_models, cv

def format_indian_number(value):
    number = int(round(float(value)))
    sign = "-" if number < 0 else ""
    digits = str(abs(number))
    if len(digits) <= 3:
        return sign + digits
    trailing = digits[-3:]
    leading = digits[:-3]
    groups = []
    while leading:
        groups.append(leading[-2:])
        leading = leading[:-2]
    return sign + ",".join([*reversed(groups), trailing])

def number_to_indian_words(value):
    ones = [
        "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
        "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
        "Seventeen", "Eighteen", "Nineteen",
    ]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

    def under_thousand(number):
        parts = []
        hundreds, remainder = divmod(number, 100)
        if hundreds:
            parts.extend([ones[hundreds], "Hundred"])
        if remainder >= 20:
            parts.append(tens[remainder // 10])
            if remainder % 10:
                parts.append(ones[remainder % 10])
        elif remainder:
            parts.append(ones[remainder])
        return " ".join(parts)

    def convert(number):
        for divisor, scale in [(10_000_000, "Crore"), (100_000, "Lakh"), (1_000, "Thousand")]:
            if number >= divisor:
                quotient, remainder = divmod(number, divisor)
                return f"{convert(quotient)} {scale}" + (f" {convert(remainder)}" if remainder else "")
        return under_thousand(number)

    number = int(round(float(value)))
    if number == 0:
        return "Zero"
    return ("Minus " if number < 0 else "") + convert(abs(number))

raw, series, featured, quality = load_project_data()
results, actual_vs_pred, train_df, test_df, final_models, cv = train_project_models(featured)

st.sidebar.title("Navigation")
page_options = ["Home", "EDA", "Model Comparison", "Forecast"]
page = st.sidebar.radio(
    "Go to",
    page_options,
    key="page_selection"
)

def navigate_to(page_name):
    st.session_state["page_selection"] = page_name

st.sidebar.markdown("---")

if page == "Home":
    st.markdown(
        """
        <style>
        .home-hero {
            text-align: center;
            padding: 0.75rem 0 3rem;
        }
        .home-hero h1 {
            font-size: 2.75rem;
            line-height: 1.15;
            margin: 0 0 0.65rem;
        }
        .home-hero h2 {
            font-size: 1.5rem;
            line-height: 1.3;
            margin: 0 0 0.65rem;
        }
        .home-hero p {
            font-size: 1.05rem;
            font-weight: 600;
            margin: 0;
        }
        </style>
        <div class="home-hero">
            <h1>Tourist Demand Forecasting</h1>
            <h2>Singapore International Visitor Arrivals</h2>
            <p>Machine Learning-Based Monthly Tourism Demand Forecasting</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project Snapshot")
    st.markdown(
        """
        <style>
        [data-testid="stMetricLabel"] {
            display: flex;
            align-items: flex-start;
            justify-content: center;
            min-height: 3.6rem;
            text-align: center;
        }
        [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] p {
            font-size: 1.5rem !important;
            font-weight: 600;
            line-height: 1.2;
            margin: 0 !important;
            text-align: center;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.4rem;
            text-align: center;
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
        }
        [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p {
            font-size: 1.4rem !important;
            line-height: 1.15;
            margin-top: 0.9rem !important;
            text-align: center;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4) [data-testid="stMetricValue"] {
            padding-top: 1.5rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4) [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p {
            margin-top: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4, gap="large", vertical_alignment="top")
    c1.metric("Dataset", "SINGSTAT", border=True, height=160)
    c2.metric("Data Period", f"{quality['start_date']} to {quality['end_date']}", border=True, height=160)
    c3.metric("Observations", f"{quality['series_rows']} monthly  \nrecords", border=True, height=160)
    c4.metric("Latest Recorded Demand", f"{int(series['Tourist_Arrivals'].iloc[-1]):,}", border=True, height=160)

    details_col1, details_col2 = st.columns(2)
    details_col1.caption("Target: International visitor arrivals")
    details_col2.caption("Frequency: Monthly   ·   Validation: Chronological split + TimeSeriesSplit")

    with st.expander("Dataset information and data quality"):
        source_col, agency_col, frequency_col = st.columns(3)
        source_col.markdown("**Dataset**\n\nInternational Visitor Arrivals by Place of Residence, Monthly")
        agency_col.markdown("**Publisher**\n\nSingapore Department of Statistics (SINGSTAT)\n\nOriginal source: Singapore Tourism Board")
        frequency_col.markdown("**Series**\n\nMonthly\n\nTarget: Tourist arrivals")

        quality_checks = [
            ("Missing target values", quality["series_missing_values"] == 0,
             f"{quality['series_missing_values']} in the prepared series"),
            ("Duplicate dates", quality["series_duplicate_dates"] == 0,
             f"{quality['series_duplicate_dates']} in the prepared series"),
            ("Date format", pd.api.types.is_datetime64_any_dtype(series["Date"]),
             "Converted to datetime"),
            ("Numeric target", pd.api.types.is_numeric_dtype(series["Tourist_Arrivals"]),
             "Arrivals converted to numeric values"),
            ("Chronological order", series["Date"].is_monotonic_increasing,
             "Dates sorted from oldest to newest"),
        ]
        checks_passed = all(check[1] for check in quality_checks)
        if checks_passed:
            st.success("Data quality status: Ready for modeling")
        else:
            st.warning("Data quality checks need attention.")
        quality_table = pd.DataFrame([
            {"Check": name, "Status": "Passed" if passed else "Review", "Details": detail}
            for name, passed, detail in quality_checks
        ])
        st.dataframe(quality_table, use_container_width=True, hide_index=True)
        st.caption(
            f"Raw file checks: {quality['raw_missing_cells']} missing cells and "
            f"{quality['raw_duplicate_rows']} duplicate rows."
        )

    st.markdown("### Machine Learning Problem")
    problem_steps = st.columns(4)
    for column, number, label in zip(
        problem_steps,
        ["01", "02", "03", "04"],
        ["Machine Learning", "Supervised Learning", "Regression", "Time-Series Forecasting"]
    ):
        column.caption(number)
        column.markdown(f"**{label}**")

    st.markdown("### Machine Learning Pipeline")
    st.markdown(
        "**CSV data** → **Cleaning** → **EDA** → **Feature engineering** → "
        "**Preprocessing** → **Model training and comparison** → **Evaluation** → **Forecast**"
    )
    st.caption("Pandas   ·   Time-series features   ·   Scikit-learn   ·   Streamlit")

    st.markdown("### Historical Tourist Demand: Singapore")
    st.caption(
        f"Monthly international visitor arrivals from {quality['start_date']} to {quality['end_date']}. "
        "Use the controls to inspect a year range or a particular month."
    )
    year_options = list(range(2008, 2027))
    filter_years, filter_month = st.columns([2, 1])
    selected_years = filter_years.select_slider(
        "Year range", options=year_options, value=(2008, 2026)
    )
    month_options = ["All months"] + list(
        pd.date_range("2020-01-01", periods=12, freq="MS").strftime("%B")
    )
    selected_month = filter_month.selectbox("Month", month_options)

    chart_series = series[
        series["Date"].dt.year.between(selected_years[0], selected_years[1])
    ]
    if selected_month != "All months":
        month_number = month_options.index(selected_month)
        chart_series = chart_series[chart_series["Date"].dt.month == month_number]

    if chart_series.empty:
        st.info("No visitor-arrival records are available for this selection.")
    else:
        fig = px.line(
            chart_series, x="Date", y="Tourist_Arrivals",
            title="Historical Tourist Demand"
        )
        covid_start = pd.Timestamp("2020-02-01")
        covid_end = pd.Timestamp("2022-03-31")
        if chart_series["Date"].min() <= covid_end and chart_series["Date"].max() >= covid_start:
            fig.add_vrect(
                x0=covid_start,
                x1=covid_end,
                fillcolor="#EF4444",
                opacity=0.14,
                line_width=0,
                annotation_text="COVID-19 impact",
                annotation_position="top left",
                annotation_font=dict(color="#F87171", size=12),
            )
        fig.update_layout(xaxis_title="Month", yaxis_title="Tourist Arrivals")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Historical Disruption")
    st.info(
        "Visitor arrivals fell exceptionally during the COVID-19 period, then recovered. "
        "This disruption is important context when interpreting model errors and forecasts."
    )

    st.markdown("### Models Compared")
    model_columns = st.columns(4)
    model_details = [
        ("Linear Regression", "Baseline", "Interpretable linear model."),
        ("Polynomial Degree 2", "Non-linear experiment", "Fits a curved time trend."),
        ("Polynomial Degree 4", "Non-linear experiment", "Tests a more flexible time trend."),
        ("Random Forest", "Ensemble", "Combines decision trees and supports feature importance."),
    ]
    for column, (model_name, role, description) in zip(model_columns, model_details):
        column.markdown(f"**{model_name}**")
        column.caption(role)
        column.write(description)

    st.markdown("### Evaluation")
    metric_columns = st.columns(4)
    for column, metric_name, explanation in zip(
        metric_columns,
        ["MAE", "MSE", "RMSE", "R²"],
        ["Mean absolute error", "Mean squared error", "Root mean squared error", "Explained variation"],
    ):
        column.markdown(f"**{metric_name}**")
        column.caption(explanation)
    st.caption("Lower MAE, MSE, and RMSE indicate smaller errors. Higher R² indicates more explained variation. TimeSeriesSplit is used for cross-validation.")

    st.markdown("### Potential Applications")
    st.write(
        "Tourism demand estimates may help inform hotel capacity, transport, workforce, "
        "marketing, and resource-planning decisions."
    )

    st.markdown("### Quick Access")
    access_columns = st.columns(3)
    quick_links = [
        ("Explore EDA", "EDA"),
        ("Compare Models", "Model Comparison"),
        ("Forecast", "Forecast"),
    ]
    for column, (label, target_page) in zip(access_columns, quick_links):
        column.button(
            label,
            on_click=navigate_to,
            args=(target_page,),
            use_container_width=True
        )

elif page == "EDA":
    st.title("Exploratory Data Analysis")

    target_name = "Tourist_Arrivals"
    target_row = raw.loc[
        raw["DataSeries"].astype(str).str.strip()
        == "Total International Visitor Arrivals By Place Of Residence"
    ]
    source_target_values = target_row.iloc[0, 1:] if not target_row.empty else pd.Series(dtype=float)
    source_target_numeric = pd.to_numeric(source_target_values, errors="coerce")
    source_target_missing = int(source_target_numeric.isna().sum())
    source_target_missing_pct = (
        source_target_missing / len(source_target_numeric) * 100
        if len(source_target_numeric) else 0.0
    )

    series_for_eda = series.copy()
    series_for_eda["Year"] = series_for_eda["Date"].dt.year
    series_for_eda["Month"] = series_for_eda["Date"].dt.month
    month_names = list(pd.date_range("2020-01-01", periods=12, freq="MS").strftime("%b"))
    series_for_eda["Month_Name"] = pd.Categorical(
        series_for_eda["Date"].dt.strftime("%b"), categories=month_names, ordered=True
    )
    display_series = series[["Date", target_name]].copy()
    display_series["Date"] = display_series["Date"].dt.strftime("%Y-%m")

    st.markdown("### 1. Dataset Overview")
    overview_columns = st.columns(4)
    overview_columns[0].metric("Observations", f"{len(series):,}")
    overview_columns[1].metric("Analysis Columns", f"{len(series.columns)}")
    overview_columns[2].metric("Date Range", f"{series['Date'].min():%Y-%m} to {series['Date'].max():%Y-%m}")
    overview_columns[3].metric("Frequency", "Monthly")
    st.caption(
        f"Target: {target_name}  ·  Prepared analysis data: {series.shape[0]:,} rows × "
        f"{series.shape[1]} columns  ·  Source CSV: {raw.shape[0]:,} rows × {raw.shape[1]:,} columns"
    )

    st.markdown("#### Column names and data types")
    dtype_table = pd.DataFrame({
        "Column": series.columns,
        "Data Type": [str(dtype) for dtype in series.dtypes],
        "Non-null Values": series.notna().sum().values,
    })
    st.dataframe(dtype_table, use_container_width=True, hide_index=True)

    first_rows, last_rows = st.columns(2)
    with first_rows:
        st.markdown("#### First 5 observations")
        st.dataframe(display_series.head(5), use_container_width=True, hide_index=True)
    with last_rows:
        st.markdown("#### Last 5 observations")
        st.dataframe(display_series.tail(5), use_container_width=True, hide_index=True)

    st.markdown("### 2. Missing Values and Duplicates")
    missing_counts = series.isna().sum()
    missing_table = pd.DataFrame({
        "Column": series.columns,
        "Missing Values": missing_counts.values,
        "Missing (%)": (missing_counts / max(len(series), 1) * 100).round(2).values,
    })
    quality_columns = st.columns(3)
    quality_columns[0].metric("Missing source target cells", f"{source_target_missing:,}")
    quality_columns[1].metric("Duplicate source rows", f"{int(raw.duplicated().sum()):,}")
    quality_columns[2].metric("Duplicate prepared dates", f"{int(series['Date'].duplicated().sum()):,}")
    st.dataframe(missing_table, use_container_width=True, hide_index=True)
    st.caption(
        f"Missing target cells in the source row: {source_target_missing:,} of "
        f"{len(source_target_numeric):,} ({source_target_missing_pct:.2f}%). "
        "The preparation step excludes invalid dates and non-numeric/missing target values; it does not impute them."
    )
    if quality["raw_missing_cells"]:
        st.caption(
            f"The full wide-format source file also contains {quality['raw_missing_cells']:,} blank cells "
            "across its other series/columns; the target-row check above isolates this analysis target."
        )

    st.markdown("### 3. Descriptive Statistics")
    descriptive_stats = series[target_name].describe().to_frame("Tourist Arrivals")
    descriptive_stats.index = [
        "Count", "Mean", "Standard deviation", "Minimum", "25th percentile",
        "Median", "75th percentile", "Maximum"
    ]
    st.dataframe(descriptive_stats, use_container_width=True)
    st.caption(
        "The mean is sensitive to unusually low or high months; compare it with the median and spread "
        "when describing typical demand."
    )

    st.markdown("### 4. Distribution and Outlier Analysis")
    target_values = series[target_name]
    first_quartile = target_values.quantile(0.25)
    third_quartile = target_values.quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    outlier_lower = first_quartile - 1.5 * interquartile_range
    outlier_upper = third_quartile + 1.5 * interquartile_range
    outlier_mask = (target_values < outlier_lower) | (target_values > outlier_upper)
    outlier_rows = series.loc[outlier_mask, ["Date", target_name]].copy()
    outlier_rows["Date"] = outlier_rows["Date"].dt.strftime("%Y-%m")
    distribution_columns = st.columns(2)
    with distribution_columns[0]:
        histogram = px.histogram(
            series, x=target_name, nbins=24, title="Monthly arrivals distribution",
            labels={target_name: "Tourist Arrivals", "count": "Months"}
        )
        st.plotly_chart(histogram, use_container_width=True)
    with distribution_columns[1]:
        distribution_box = px.box(
            series, y=target_name, points="outliers", title="Overall arrivals box plot",
            labels={target_name: "Tourist Arrivals"}
        )
        st.plotly_chart(distribution_box, use_container_width=True)
    st.write(
        f"The 1.5×IQR rule flags **{len(outlier_rows)} candidate outlier months** "
        f"(below {outlier_lower:,.0f} or above {outlier_upper:,.0f}). These are candidates for investigation, "
        "not automatic errors or observations to remove."
    )
    if not outlier_rows.empty:
        st.dataframe(outlier_rows, use_container_width=True, hide_index=True)

    st.markdown("### 5. Time-Series Trend and Historical Demand")
    trend_index = np.arange(len(series))
    trend_slope, trend_intercept = np.polyfit(trend_index, target_values.to_numpy(), 1)
    trend_series = series[["Date", target_name]].copy()
    trend_series["Linear Trend"] = trend_intercept + trend_slope * trend_index
    trend_chart = px.line(
        trend_series, x="Date", y=[target_name, "Linear Trend"],
        title="Monthly arrivals and fitted linear trend"
    )
    covid_start = pd.Timestamp("2020-02-01")
    covid_end = pd.Timestamp("2022-03-31")
    if series["Date"].min() <= covid_end and series["Date"].max() >= covid_start:
        trend_chart.add_vrect(
            x0=covid_start, x1=covid_end, fillcolor="#EF4444", opacity=0.12,
            line_width=0, annotation_text="COVID-19 disruption",
            annotation_position="top left", annotation_font=dict(color="#F87171", size=11)
        )
    trend_chart.update_layout(xaxis_title="Month", yaxis_title="Tourist Arrivals")
    st.plotly_chart(trend_chart, use_container_width=True)
    st.caption(
        f"The fitted linear slope is {trend_slope:,.0f} arrivals per observed month. "
        "It summarizes the full period and should be interpreted alongside the COVID-era shock and recovery."
    )

    st.markdown("### 6. Year-over-Year Analysis")
    yearly = series_for_eda.groupby("Year", as_index=False).agg(
        Annual_Total=(target_name, "sum"),
        Monthly_Average=(target_name, "mean"),
        Months_Observed=(target_name, "count"),
    )
    yearly["Year-over-Year Change (%)"] = yearly["Annual_Total"].pct_change().mul(100).round(2)
    yearly_chart = px.bar(
        yearly, x="Year", y="Annual_Total", title="Annual visitor arrivals",
        labels={"Annual_Total": "Total Arrivals", "Year": "Year"},
        hover_data=["Monthly_Average", "Months_Observed"]
    )
    st.plotly_chart(yearly_chart, use_container_width=True)
    latest_year = int(yearly["Year"].iloc[-1])
    latest_year_months = int(yearly["Months_Observed"].iloc[-1])
    st.caption(
        f"Annual totals are calculated from observed months. {latest_year} currently contains "
        f"{latest_year_months} month(s), so it is a partial-year total and is not directly comparable with full years."
    )
    st.dataframe(yearly.round(2), use_container_width=True, hide_index=True)

    st.markdown("### 7. Monthly Seasonality")
    monthly_summary = series_for_eda.groupby("Month_Name", observed=False)[target_name].agg(
        Mean="mean", Median="median", Standard_Deviation="std", Observations="count"
    ).reset_index()
    monthly_summary["Month_Name"] = monthly_summary["Month_Name"].astype(str)
    monthly_chart = px.bar(
        monthly_summary, x="Month_Name", y="Mean", title="Average arrivals by calendar month",
        labels={"Month_Name": "Month", "Mean": "Average Arrivals"},
        category_orders={"Month_Name": month_names}
    )
    st.plotly_chart(monthly_chart, use_container_width=True)
    monthly_box = px.box(
        series_for_eda, x="Month_Name", y=target_name, points="outliers",
        title="Arrivals distribution by calendar month",
        labels={"Month_Name": "Month", target_name: "Tourist Arrivals"},
        category_orders={"Month_Name": month_names}
    )
    st.plotly_chart(monthly_box, use_container_width=True)
    st.dataframe(monthly_summary.round(2), use_container_width=True, hide_index=True)

    st.markdown("### 8. Year × Month Seasonality Heatmap")
    heatmap_data = series_for_eda.pivot_table(
        index="Year", columns="Month", values=target_name, aggfunc="mean"
    ).reindex(columns=range(1, 13))
    heatmap_data.columns = month_names
    heatmap = px.imshow(
        heatmap_data, aspect="auto", color_continuous_scale="YlOrRd",
        labels={"x": "Month", "y": "Year", "color": "Arrivals"},
        title="Monthly visitor arrivals by year"
    )
    st.plotly_chart(heatmap, use_container_width=True)
    st.caption("Blank cells indicate months not present in the source data; color encodes arrivals.")

    st.markdown("### 9. Rolling Averages")
    rolling_series = series[["Date", target_name]].copy()
    rolling_series["3-month rolling mean"] = target_values.rolling(3).mean()
    rolling_series["12-month rolling mean"] = target_values.rolling(12).mean()
    rolling_chart = px.line(
        rolling_series, x="Date", y=[target_name, "3-month rolling mean", "12-month rolling mean"],
        title="Observed arrivals with 3- and 12-month rolling means"
    )
    st.plotly_chart(rolling_chart, use_container_width=True)
    st.caption("Rolling means smooth short-term variation; early values are blank until each window is available.")

    st.markdown("### 10. Lag Relationships and Autocorrelation")
    lag_columns = st.columns(2)
    with lag_columns[0]:
        lag_one_chart = px.scatter(
            featured, x="Lag_1", y=target_name, title="Current arrivals vs previous month",
            labels={"Lag_1": "Previous month's arrivals", target_name: "Current arrivals"},
            trendline=None
        )
        st.plotly_chart(lag_one_chart, use_container_width=True)
    with lag_columns[1]:
        lag_twelve_chart = px.scatter(
            featured, x="Lag_12", y=target_name, title="Current arrivals vs same month last year",
            labels={"Lag_12": "Arrivals 12 months earlier", target_name: "Current arrivals"},
            trendline=None
        )
        st.plotly_chart(lag_twelve_chart, use_container_width=True)

    lag_correlations = {
        lag: float(target_values.autocorr(lag=lag))
        for lag in range(0, min(25, len(target_values)))
    }
    acf_data = pd.DataFrame({"Lag (months)": list(lag_correlations), "Correlation": list(lag_correlations.values())})
    acf_chart = px.bar(
        acf_data, x="Lag (months)", y="Correlation", title="Autocorrelation by monthly lag",
        labels={"Correlation": "Autocorrelation"}
    )
    confidence_bound = 1.96 / np.sqrt(len(target_values))
    acf_chart.add_hline(y=confidence_bound, line_dash="dash", line_color="#F87171")
    acf_chart.add_hline(y=-confidence_bound, line_dash="dash", line_color="#F87171")
    st.plotly_chart(acf_chart, use_container_width=True)
    lag_one_correlation = lag_correlations.get(1, float("nan"))
    lag_twelve_correlation = lag_correlations.get(12, float("nan"))
    st.caption(
        f"Descriptive autocorrelation: lag 1 = {lag_one_correlation:.3f}; "
        f"lag 12 = {lag_twelve_correlation:.3f}. Dashed lines show the approximate ±1.96/√n reference; "
        "these are exploratory guides, not a formal model significance test."
    )

    st.markdown("### 11. Feature Relationships")
    correlation_columns = [
        target_name, "Lag_1", "Lag_12", "Rolling_Mean_3", "Month_Sin", "Month_Cos"
    ]
    feature_correlations = featured[correlation_columns].corr()
    correlation_chart = px.imshow(
        feature_correlations, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
        color_continuous_scale="RdBu_r", title="Target and selected feature correlations"
    )
    st.plotly_chart(correlation_chart, use_container_width=True)
    st.caption("Correlations describe pairwise linear association and do not establish causation.")

    st.markdown("### 12. EDA Findings and Feature-Engineering Rationale")
    first_year_average = target_values.head(12).mean()
    latest_year_average = target_values.tail(12).mean()
    period_change = (latest_year_average / first_year_average - 1) * 100 if first_year_average else 0.0
    highest_month = monthly_summary.loc[monthly_summary["Mean"].idxmax()]
    lowest_month = monthly_summary.loc[monthly_summary["Mean"].idxmin()]
    pre_covid = series.loc[series["Date"].between("2017-01-01", "2019-12-31"), target_name]
    covid_period = series.loc[series["Date"].between(covid_start, covid_end), target_name]
    if len(pre_covid) and len(covid_period) and pre_covid.mean():
        covid_change = (covid_period.mean() / pre_covid.mean() - 1) * 100
        covid_finding = (
            f"Average arrivals during Feb 2020–Mar 2022 were {abs(covid_change):.1f}% "
            f"{'lower' if covid_change < 0 else 'higher'} than the 2017–2019 average. "
            "This real-world disruption is retained, not treated as an automatic data error."
        )
    else:
        covid_finding = "The selected dataset does not contain both comparison periods needed to quantify the COVID-era change."

    st.markdown(
        f"- The mean of the latest 12 observations is **{period_change:+.1f}%** relative to the first 12, "
        "summarizing long-term change while averaging over individual months.\n"
        f"- The highest average calendar month is **{highest_month['Month_Name']}** "
        f"({highest_month['Mean']:,.0f}); the lowest is **{lowest_month['Month_Name']}** "
        f"({lowest_month['Mean']:,.0f}).\n"
        f"- {covid_finding}\n"
        f"- Lag correlations are **{lag_one_correlation:.3f}** at one month and "
        f"**{lag_twelve_correlation:.3f}** at 12 months; these help assess whether recent and annual history "
        "carry useful descriptive signal.\n"
        f"- The IQR rule flags **{len(outlier_rows)}** candidate months. Investigate context before deciding "
        "whether any value is erroneous; the EDA does not remove them."
    )

    feature_rationale = pd.DataFrame({
        "EDA evidence": [
            f"Month averages range from {lowest_month['Mean']:,.0f} to {highest_month['Mean']:,.0f} arrivals.",
            f"One-month autocorrelation is {lag_one_correlation:.3f}.",
            f"Twelve-month autocorrelation is {lag_twelve_correlation:.3f}.",
            "Rolling means provide a smoothed summary of recent levels.",
        ],
        "Related model features": [
            "Month_Sin, Month_Cos, Month, Quarter",
            "Lag_1, Lag_2, Lag_3",
            "Lag_12",
            "Rolling_Mean_3, Rolling_Mean_6, Rolling_Mean_12",
        ],
    })
    st.dataframe(feature_rationale, use_container_width=True, hide_index=True)
    st.info(
        "The baseline, polynomial trend experiments, and Random Forest are compared because the series "
        "contains changing trend, monthly patterns, lag structure, and a major external shock. "
        "Chronological test results and TimeSeriesSplit—not EDA alone—determine comparative model performance."
    )

elif page == "Model Comparison":
    st.title("Model Development & Evaluation")

    st.markdown("### Test-set model comparison")
    display_results = results.copy()
    for c in ["MAE", "MSE", "RMSE"]:
        display_results[c] = display_results[c].round(2)
    display_results["R2"] = display_results["R2"].round(4)
    st.dataframe(display_results, use_container_width=True, hide_index=True)

    st.markdown("### Actual vs Predicted")
    model_choice = st.selectbox(
        "Choose a model",
        ["Linear Regression", "Random Forest",
         "Polynomial Regression (Degree 2)", "Polynomial Regression (Degree 4)"]
    )
    plot_df = actual_vs_pred[["Date", "Tourist_Arrivals", model_choice]].copy()
    plot_df = plot_df.rename(columns={model_choice: "Predicted"})
    fig = px.line(plot_df, x="Date", y=["Tourist_Arrivals", "Predicted"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Random Forest Feature Importance")
    rf_imp = random_forest_feature_importance(final_models["Random Forest"]).head(10)
    fig_imp = px.bar(rf_imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### Time-Series Cross-Validation")
    cv_show = cv.copy()
    for c in ["MAE", "MSE", "RMSE"]:
        cv_show[c] = cv_show[c].round(2)
    cv_show["R2"] = cv_show["R2"].round(4)
    st.dataframe(cv_show, use_container_width=True, hide_index=True)

    summary = cv.groupby("Model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).round(2)
    st.markdown("### Cross-validation summary")
    st.dataframe(summary, use_container_width=True)

elif page == "Forecast":
    st.title("Future Tourist Demand Forecast")

    c1, c2 = st.columns([1, 2])
    model_name = c1.selectbox("Forecast Model", ["Linear Regression", "Random Forest"])
    forecast_mode = c2.radio(
        "Forecast selection",
        ["Next N months", "Choose specific month(s)"],
        index=1,
        key="forecast_selection_mode",
        horizontal=True,
    )
    last_observed_date = series["Date"].max()
    available_forecast_dates = pd.date_range(
        last_observed_date + pd.DateOffset(months=1), periods=12, freq="MS"
    )
    month_labels = {
        date.strftime("%Y-%m"): f"{date.strftime('%B')} (Month {date.month:02d}, {date.year})"
        for date in available_forecast_dates
    }
    date_by_month_label = {
        label: pd.Timestamp(month_key + "-01")
        for month_key, label in month_labels.items()
    }
    selected_month_labels = []
    if forecast_mode == "Next N months":
        horizon = c2.slider("Forecast Horizon (months)", 1, 12, 6)
    else:
        selected_month_labels = c2.multiselect(
            "Select target month(s)", list(month_labels.values())
        )
        c2.caption(
            "Forecasts are generated sequentially; unselected months are modeled internally "
            "when needed to reach a later selected month."
        )

    st.caption(
        "Forecasting is recursive: after the first predicted month, the prediction is used "
        "as historical input for later months."
    )

    if st.button("Generate Forecast", type="primary"):
        if forecast_mode == "Choose specific month(s)" and not selected_month_labels:
            st.warning("Choose at least one target month before generating a forecast.")
        else:
            selected_dates = [date_by_month_label[label] for label in selected_month_labels]
            forecast_horizon = horizon if forecast_mode == "Next N months" else max(
                (date.year - last_observed_date.year) * 12 + date.month - last_observed_date.month
                for date in selected_dates
            )
            forecast_path = recursive_forecast(
                final_models[model_name], series, months=forecast_horizon
            )
            forecast = forecast_path
            if forecast_mode == "Choose specific month(s)":
                forecast = forecast[forecast["Date"].isin(selected_dates)].reset_index(drop=True)
            st.session_state["forecast"] = forecast
            st.session_state["forecast_path"] = forecast_path
            st.session_state["forecast_selected_dates"] = selected_dates
            st.session_state["forecast_model"] = model_name
            st.session_state["forecast_mode"] = forecast_mode

    if "forecast" in st.session_state:
        forecast = st.session_state["forecast"]
        st.success(f"Forecast generated using {st.session_state['forecast_model']}.")
        forecast_table = pd.DataFrame({
            "Forecast Month": forecast["Date"].dt.strftime("%B (Month %m), %Y"),
            "Predicted Arrivals": forecast["Predicted_Tourist_Arrivals"].map(format_indian_number),
            "Arrivals in Words": forecast["Predicted_Tourist_Arrivals"].map(number_to_indian_words),
        })
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)

        history_plot = series.tail(36)[["Date", "Tourist_Arrivals"]].copy()
        history_plot["Type"] = "Historical"
        forecast_path = st.session_state.get("forecast_path", forecast)
        forecast_path_plot = forecast_path.rename(
            columns={"Predicted_Tourist_Arrivals": "Tourist_Arrivals"}
        )[["Date", "Tourist_Arrivals"]].copy()
        forecast_path_plot["Type"] = "Forecast path"
        forecast_anchor = history_plot.tail(1).copy()
        forecast_anchor["Type"] = "Forecast path"
        forecast_path_plot = pd.concat(
            [forecast_anchor, forecast_path_plot], ignore_index=True
        )
        combined = pd.concat([history_plot, forecast_path_plot], ignore_index=True)

        fig = px.line(
            combined, x="Date", y="Tourist_Arrivals", color="Type",
            title="Recent History + Forecast Path", markers=True,
            color_discrete_map={
                "Historical": "#82C8FF",
                "Forecast path": "#0B84F3",
            }
        )
        selected_forecast_dates = st.session_state.get("forecast_selected_dates", [])
        selected_forecast_points = forecast_path[
            forecast_path["Date"].isin(selected_forecast_dates)
        ]
        if not selected_forecast_points.empty:
            fig.add_scatter(
                x=selected_forecast_points["Date"],
                y=selected_forecast_points["Predicted_Tourist_Arrivals"],
                mode="markers",
                name="Selected target month(s)",
                marker=dict(color="#FF6B6B", size=11, symbol="diamond"),
                hovertemplate="%{x|%B %Y}<br>Selected forecast: %{y:,.0f}<extra></extra>",
            )
        fig.update_layout(xaxis_title="Date", yaxis_title="Tourist Arrivals")
        st.plotly_chart(fig, use_container_width=True)
        if selected_forecast_dates:
            st.caption(
                "The blue line shows every recursively forecast month; red diamonds mark your selected target month(s)."
            )

        total_forecast = forecast["Predicted_Tourist_Arrivals"].sum()
        avg_forecast = forecast["Predicted_Tourist_Arrivals"].mean()
        a, b = st.columns(2)
        if len(forecast) == 1:
            selected_month = forecast["Date"].iloc[0]
            a.metric(
                "Selected Month Forecast",
                format_indian_number(forecast["Predicted_Tourist_Arrivals"].iloc[0])
            )
            a.caption(
                f"In words: {number_to_indian_words(forecast['Predicted_Tourist_Arrivals'].iloc[0])} visitors"
            )
            b.metric("Forecast Month", selected_month.strftime("%B (Month %m), %Y"))
        else:
            a.metric("Forecast Total", format_indian_number(total_forecast))
            a.caption(f"In words: {number_to_indian_words(total_forecast)} visitors")
            b.metric("Average Monthly Forecast", format_indian_number(avg_forecast))
            b.caption(f"In words: {number_to_indian_words(avg_forecast)} visitors")

