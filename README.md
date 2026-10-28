# EV Charging Demand Prediction and Usage Pattern Analysis

An end-to-end machine learning project on EV charging sessions: data cleaning, EDA,
feature engineering, regression (predict charging demand), classification (High vs Normal
demand), class-imbalance experiments, SHAP and LIME explanations, daily demand
forecasting, and an interactive Streamlit dashboard.

## Problem statement

Predict EV charging demand from charger type, location, charging company, time of
charging and historical usage, compare several models, handle class imbalance, and
explain which factors drive demand.

## Dataset

- Public EV charging transaction dataset ("Charging Record", link provided with the brief).
- 72,856 raw sessions, 13 columns; 72,826 sessions after cleaning.
- Period: 30 Sep 2021 to 30 Sep 2022. 2,118 chargers, 2,336 identified users, 14 location types.
- `Demand` is the regression target (assumed kWh); High Demand means Demand at or above
  the training-set median (14.2).
- The raw file is not committed. Place it at `data/raw/charging_record.csv`.
  Cleaned and split data are in `data/processed/`.

## Approach

| Step | Notebook | What happens |
|---|---|---|
| 1 | `01_data_understanding` | Profile the data, find quality issues, assess leakage risk |
| 2 | `02_data_cleaning` | Remove duplicates and impossible sessions, build datetimes, flag inconsistent durations |
| 3 | `03_eda` | Patterns by location, charger type, company, hour, weekday, trend |
| 4 | `04_feature_engineering` | Time features, leak-free history features, chronological split |
| 5 | `05_regression` | Baseline plus 6 models, MAE/MSE/RMSE/R2, leakage experiment |
| 6 | `06_feature_importance` | Gain and permutation importance |
| 7 | `07_time_series_forecast` | Rolling-backtest of 5 daily-demand forecasting methods |
| 8 | `08_classification` | 7 classifiers plus baseline, imbalance experiment |
| 9 | `09_explainability` | SHAP and LIME, saved artifacts |

### Design decisions that matter

- **No data leakage.** `Duration` and end times are unknown when a session starts, so they are
  excluded. Adding Duration lifts R2 from 0.35 to 0.79, which shows why.
- **Chronological split.** Train on the first 80% of time (before 30 Jul 2022), test on the last 20%.
  Medians and fill values come from training data only.
- **History features use only earlier sessions** (`shift()` before averaging), for chargers and for
  identified users. `UserID` 0 is treated as "unknown user".
- **Duration is not trusted.** About 2,200 sessions disagree with their own timestamps, so a
  recomputed duration was added and the originals flagged.

## Results

### Regression (test set, predicting Demand)

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| LightGBM | 8.159 | 11.212 | 0.351 |
| Random Forest | 8.198 | 11.246 | 0.347 |
| Gradient Boosting | 8.465 | 11.408 | 0.328 |
| XGBoost | 8.559 | 11.479 | 0.319 |
| Decision Tree | 8.419 | 11.490 | 0.318 |
| Linear Regression | 8.568 | 11.525 | 0.314 |
| Baseline (mean) | 10.865 | 13.918 | -0.001 |

The best model is about 25% better than the baseline on MAE. The models are close; most of the
gain comes from the features, not the algorithm.

### Classification (High vs Normal Demand)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LightGBM | 0.728 | 0.694 | 0.790 | 0.739 | 0.806 |
| XGBoost | 0.726 | 0.691 | 0.793 | 0.739 | 0.806 |
| Random Forest | 0.725 | 0.693 | 0.783 | 0.735 | 0.805 |
| Decision Tree | 0.708 | 0.691 | 0.728 | 0.709 | 0.789 |
| Logistic Regression | 0.706 | 0.693 | 0.712 | 0.702 | 0.787 |
| SVM (15,000-row sample) | 0.717 | 0.684 | 0.781 | 0.729 | 0.784 |
| KNN | 0.704 | 0.681 | 0.741 | 0.710 | 0.777 |

### Class imbalance

The median split is balanced by construction (about 50/50), so resampling is not needed for the
brief's target. To demonstrate the techniques, a stricter "peak demand" target (top 10%,
threshold 37.2) was used:

| Method | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| No resampling | 0.899 | 0.594 | 0.265 | 0.367 | 0.862 |
| class_weight balanced | 0.781 | 0.302 | 0.753 | 0.431 | 0.863 |
| Random oversampling | 0.784 | 0.302 | 0.741 | 0.429 | 0.859 |
| SMOTE | 0.765 | 0.286 | 0.760 | 0.416 | 0.847 |
| ADASYN | 0.782 | 0.299 | 0.735 | 0.425 | 0.848 |

Accuracy is misleading here: always guessing "not peak" is about 89% accurate. Resampling raised
recall (0.27 to about 0.75) at the cost of precision, but did not improve ROC-AUC.

### Time-series forecast (daily total demand, 12 weekly backtests, 7-day horizon)

| Method | MAE (kWh/day) | MAPE |
|---|---|---|
| Moving average (7 days) | 488.1 | 16.6% |
| Seasonal average (4 weeks) | 495.3 | 17.8% |
| Trend + weekday regression | 530.8 | 19.0% |
| Seasonal naive | 562.7 | 19.4% |
| Naive | 581.6 | 17.3% |

## Key findings

1. A user's own past average demand is by far the strongest predictor, followed by the charger's
   past average and the start hour (SHAP, permutation and LIME agree).
2. Demand per session is highest for sessions starting late evening and overnight; session
   volume peaks at 17:00-18:00.
3. About half of chargers have fewer than 10 sessions; the busiest 10% handle about 63%.
4. Forecast errors are about 14-17% of average daily demand; simple methods are hard to beat.

## Dashboard

Five tabs: Overview, Demand Patterns, Forecast, Predict a Session (with live SHAP and optional
LIME), and Model Insights. Sidebar filters: date range, location, charger type, company.

## Setup

Requires Python 3.12+ (developed on 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
pip install -r requirements.txt
```

1. Place `charging_record.csv` in `data/raw/`.
2. Run notebooks `01` to `09` in order (they write to `data/processed/`, `models/`, `docs/figures/`).
3. Check the setup: `python tests/test_project.py`
4. Launch: `streamlit run app/app.py`

## Project structure

```
app/            Streamlit dashboard
src/            config, data, predict, explain modules
tests/          checks that dashboard features match training
notebooks/      01-09 analysis notebooks
data/processed/ cleaned and split datasets
models/         trained models, results tables, forecasts
docs/           data dictionary, saved figures
```

## Limitations

- Only one year of data, so annual seasonality cannot be learned and August never appears in training.
- The meaning of `ChargerCompany` and `ChargerType` codes (0/1) is not documented, so they are
  treated as anonymous categories.
- An apparent outage on 5 Sep 2022 (42 sessions vs about 220 normal) was kept in the data.
- Predictions are moderate (R2 about 0.35); demand also depends on battery state and stay time,
  which are unknown in advance.
- SMOTE interpolates between sessions and can create in-between values in 0/1 columns.
- In the dashboard, session history is typed in; a real system would look it up.

## Author

Your name