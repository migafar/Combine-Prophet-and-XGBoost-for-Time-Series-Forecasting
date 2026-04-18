# Combine Prophet and XGBoost for Time Series Forecasting: Prophet + XGBoost

A demand forecasting system that combines Prophet for trend and seasonality modelling with XGBoost for residual correction using external regressors.

---

## Overview

The core idea is a two-stage pipeline:

1. **Prophet** fits the structural components of the time series — trend, weekly seasonality, yearly seasonality, and holidays.
2. **XGBoost** learns the residuals (errors) left by Prophet, using external features like temperature and promotions.
3. The **hybrid forecast** is their sum: `prophet_pred + xgb_residual_pred`.

```
Actual = Trend + Seasonality + External Effects + Noise
          |______ Prophet _____|   |_____ XGBoost _____|
```

---

## Project Structure

```
project/
├── Code   # Main notebook
└── README.md
```

---

## Installation

```bash
pip install prophet xgboost scikit-learn pandas numpy matplotlib seaborn
```

Then open the notebook:

```bash
jupyter notebook Combine Prophet and XGBoost for Time Series Forecasting.ipynb
```

---

## Dataset

The notebook uses a synthetic retail demand dataset spanning 730 days (2 years). It is generated programmatically and includes the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `ds` | datetime | Date |
| `y` | float | Daily demand in units |
| `temperature` | float | Air temperature in Celsius |
| `promotion` | int (0/1) | Whether a promotion was active |
| `holiday` | int (0/1) | Whether it was a holiday |

---

## Model Architecture

```
Input Data (date, demand, temperature, promotion, holiday)
        |
        v
    Prophet
    - Piecewise linear trend
    - Weekly and yearly seasonality
    - Holiday effects
    - External regressors: temperature, promotion
        |
        v
    Residuals = actual - prophet_pred
        |
        v
    XGBoost
    - Trained on residuals
    - Features: temperature, promotion, holiday,
                day_of_week, month, day_of_year,
                week_of_year, is_weekend, prophet_pred
        |
        v
    Hybrid Forecast = prophet_pred + xgb_residual_pred
```

---

## Notebook Walkthrough

| Step | Section | Description |
|------|---------|-------------|
| 1 | Imports | Load all required libraries |
| 2 | Data Generation | Create synthetic demand dataset |
| 3 | EDA | Demand series, temperature scatter, day-of-week averages |
| 4 | Train/Test Split | Last 60 days held out for testing |
| 5 | Prophet | Fit baseline model on trend and seasonality |
| 6 | XGBoost | Fit residual model with external features |
| 7 | Hybrid Forecast | Combine both model outputs |
| 8 | Metrics | Compute MAPE and MAE for both models |
| 9 | Bar Chart | Visual comparison of model performance |
| 10 | Forecast Plot | Actual vs Prophet vs Hybrid on test set |
| 11 | Residual Analysis | Assess XGBoost correction quality |
| 12 | Summary | Final results and improvement percentage |

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| MAPE | `mean(|actual - pred| / actual) * 100` | Percentage error |
| MAE | `mean(|actual - pred|)` | Average absolute error in units |

---

## Expected Results

```
Model                       MAPE        MAE
--------------------------------------------
Prophet Baseline            8 - 12 %    20 - 30 units
Hybrid (Prophet + XGBoost)  5 - 8  %    12 - 20 units
--------------------------------------------
Improvement                 ~30 - 40 %  ~25 - 35 %
```

The hybrid model consistently outperforms Prophet alone because XGBoost captures non-linear relationships between external variables and residual errors that Prophet cannot model.

---

## Key Takeaways

- Prophet is well-suited for capturing interpretable time series structure but struggles with irregular external effects.
- XGBoost is flexible but lacks built-in handling of trend and seasonality.
- Combining them produces a model that is both interpretable and accurate.

---

## Requirements

| Package | Purpose |
|---------|---------|
| `prophet` | Trend and seasonality modelling |
| `xgboost` | Residual correction with external features |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `scikit-learn` | Evaluation metrics |
