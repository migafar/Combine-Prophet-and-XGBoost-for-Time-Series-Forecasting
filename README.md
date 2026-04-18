# Combine Prophet and XGBoost for Time Series Forecasting

## What is this project about?

Forecasting demand is tricky. Real-world demand does not follow a clean pattern — it is shaped by long-term trends, seasonal cycles, sudden promotions, weather changes, and holidays all at once. A single model rarely handles all of these well on its own.

This project takes a hybrid approach: instead of forcing one model to do everything, we split the problem in two. Prophet handles the structural side of the time series — the trend and seasonality — while XGBoost steps in to correct what Prophet misses, using external signals like temperature and promotional activity. The final forecast is simply the sum of both.

---

## How it works

Think of it as a two-pass system.

In the first pass, Prophet looks at the historical demand and learns its shape — how demand grows over time, how it dips on weekdays and peaks on weekends, how it follows a yearly pattern tied to seasons. It also accounts for known holidays. This gives us a solid baseline forecast, but it is not perfect. Prophet is good at structure, not at reacting to irregular external events.

That is where the second pass comes in. We take the errors that Prophet made on the training data — called residuals — and feed them into XGBoost along with features like temperature, day of the week, and whether a promotion was running. XGBoost learns to predict those leftover errors. When we add its predictions back onto Prophet's forecast, the result is more accurate than either model would produce alone.

```
Actual = Trend + Seasonality + External Effects + Noise
          |______ Prophet _____|   |_____ XGBoost _____|
```

---

## Project Structure

```
project/
├── Code        # Main notebook with all steps
└── README.md
```

---

## The Dataset

Rather than relying on a real dataset that may not be publicly available, the notebook generates its own synthetic retail demand data covering two years (730 days). This makes it easy to run, reproduce, and experiment with.

The dataset includes five columns:

| Column | Type | Description |
|--------|------|-------------|
| `ds` | datetime | The date of each observation |
| `y` | float | Daily demand measured in units |
| `temperature` | float | Air temperature in Celsius |
| `promotion` | int (0/1) | Whether a promotion was active that day |
| `holiday` | int (0/1) | Whether it was a public holiday |

The demand signal is built from a realistic mix of trend, seasonality, temperature effects, promotional lifts, and random noise — so the models have something meaningful to learn from.

---

## Model Architecture

Here is how data flows through the system:

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

## What the Notebook Covers

The notebook walks through the full pipeline in twelve steps, starting from raw data and ending with a comparison between the Prophet baseline and the hybrid model.

| Step | Section | Description |
|------|---------|-------------|
| 1 | Imports | Load all required libraries |
| 2 | Data Generation | Create the synthetic demand dataset |
| 3 | EDA | Explore the demand series, temperature relationship, and day-of-week patterns |
| 4 | Train/Test Split | Hold out the last 60 days for evaluation |
| 5 | Prophet | Fit the baseline model on trend and seasonality |
| 6 | XGBoost | Fit the residual model using external features |
| 7 | Hybrid Forecast | Combine both model outputs into a single forecast |
| 8 | Metrics | Compute MAPE and MAE for both models |
| 9 | Bar Chart | Visually compare model performance |
| 10 | Forecast Plot | Plot actual vs Prophet vs Hybrid on the test period |
| 11 | Residual Analysis | Check how well XGBoost corrected Prophet's errors |
| 12 | Summary | Review the final numbers and overall improvement |

---

## How We Measure Performance

Two metrics are used to evaluate both models on the 60-day test set:

| Metric | Formula | What it tells you |
|--------|---------|---------|
| MAPE | `mean(|actual - pred| / actual) * 100` | How far off the forecast is, expressed as a percentage |
| MAE | `mean(|actual - pred|)` | The average error in absolute units |

Lower is better for both.

---

## What Results to Expect

Running the notebook should produce something close to this:

```
Model                       MAPE        MAE
--------------------------------------------
Prophet Baseline            8 - 12 %    20 - 30 units
Hybrid (Prophet + XGBoost)  5 - 8  %    12 - 20 units
--------------------------------------------
Improvement                 ~30 - 40 %  ~25 - 35 %
```

The hybrid model wins because XGBoost picks up on non-linear patterns in the residuals that Prophet simply has no way to model — things like a promotion happening during a heatwave having a bigger lift than either factor would produce on its own.

---

## Key Takeaways

Prophet is a strong starting point for any time series with clear trend and seasonal structure. It is interpretable, fast to train, and handles holidays out of the box. But it was not designed to react dynamically to external variables.

XGBoost fills that gap. It is not aware of time structure by default, so using it on raw demand would miss the trend and seasonality entirely. But on top of Prophet's residuals, it has exactly the right problem to solve.

Together, the two models cover each other's weaknesses. That is the core lesson of this project: combining simple, well-understood models often beats building one complicated model that tries to do everything.

---

## Requirements

| Package | Purpose |
|---------|---------|
| `prophet` | Trend and seasonality modelling |
| `xgboost` | Residual correction with external features |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and visualisation |
| `scikit-learn` | Evaluation metrics |

Install everything with:

```bash
pip install prophet xgboost scikit-learn pandas numpy matplotlib seaborn
```
