# Data Understanding: EV Charging Records

**Size:** 72,856 sessions × 13 columns. **Period:** 2021-09-30 to 2022-09-30 (one year).
**Unit of analysis:** one row = one charging session.

## Column dictionary

| Column | Type | Meaning | Known when the session starts? |
|---|---|---|---|
| UserID | Integer ID | Who charged. **0 appears to mean an unidentified user** (31,553 rows, 43%). 2,336 real users remain. | Yes |
| ChargerID | Integer ID | Which charger (2,119 chargers). Very uneven: median 9 sessions, max 674. | Yes |
| ChargerCompany | Category (0/1) | Charging company. 1 = 45,713 sessions, 0 = 27,143. | Yes |
| Location | Category (14 values) | Where the charger is. Largest: public area, apartment, resort, hotel. Smallest: bus garage (29), accommodation (187). | Yes |
| ChargerType | Category (0/1) | Charger type. 0 = 57,903, 1 = 14,953. The codes are not defined in the brief. | Yes |
| StartDay, StartTime | Text | Session start date and time (with seconds). | Yes |
| StartDatetime | Text | Start date and time to the minute. Convert to datetime. | Yes |
| EndDay, EndTime, EndDatetime | Text | Session end. | **No** |
| Duration | Integer | **Minutes** plugged in. It equals EndDatetime minus StartDatetime exactly. | **No** |
| Demand | Float | Energy delivered, assumed kWh (median 14.1, max 97). **The target.** | **No** |

## Findings about the data

**Good news**
- No missing values in any column.
- All dates parse correctly.
- Each charger has exactly one company, one location and one type, so those three describe the charger, not the session.

**Problems to fix **
- **13 duplicate rows.**
- **17 sessions end before they start** (negative Duration, e.g. -29). Both timestamps are untrustworthy, and they are 0.02% of the data, so we will drop them.
- **718 sessions have Duration = 0.** Check whether they have real Demand before deciding what to do.
- **Date columns are text,** and need converting to datetime.
- **ID columns look like numbers but are labels.** Averaging UserID is meaningless.

**Distribution notes**
- Demand is right-skewed: median 14.1, mean 17.4, 99th percentile 58.9, max 97. No extreme errors.
- Duration is also right-skewed: median 112 min, 99th percentile 634, max 1,573 (about 26 hours).
- ChargerCompany 1 only uses ChargerType 0. Company 0 uses both types.
- Mean demand: type 0 = 17.05, type 1 = 18.96.

## Decisions that affect the modelling

1. **Leakage risk.** EndTime, EndDatetime and Duration are not known when a session starts. Demand vs Duration correlation is 0.66. That is moderate, so it will boost scores noticeably, but it is not a real prediction. The main models exclude these columns. An optional comparison run will show the effect.
2. **UserID = 0 must be excluded from user-history features.** Treating 31,553 unrelated sessions as one "heavy user" would create a fake history.
3. **Classification target.** High Demand means Demand ≥ 14.1. This gives 50.05% High and 49.95% Normal, which is balanced by construction. 58 sessions sit exactly on the median and fall into High. SMOTE/ADASYN are not needed for this target, so we will state that honestly and demonstrate them on a stricter optional "peak" target.
4. **High-cardinality IDs.** ChargerID (2,119 values) and UserID (2,337 values) cannot be one-hot encoded sensibly. We will use history-based features, such as a charger's past average demand, built so they only use earlier sessions.

## Cleaning summary
- Removed 13 duplicates and 17 sessions ending before they start.
- Built datetimes from Day + Time to keep seconds.
- Recorded Duration disagrees with timestamps in about 2,200 rows;
  added DurationCalc and a flag instead of deleting those rows.
- Outliers reported but kept (real long sessions).
- UserID 0 flagged as IsKnownUser = False.
- Data sorted by StartDatetime and saved to data/processed/clean_sessions.csv.

## Feature engineering summary
- Time features: Hour, Day, Month, DayOfWeek, IsWeekend, TimeOfDay.
- History features (charger and user): count and average demand of strictly earlier sessions.
- Unknown users (UserID 0) get no user history.
- Chronological 80/20 split at 2022-07-30; thresholds and fill values computed on train only.
- Excluded for leakage: Duration, DurationCalc, EndDatetime (unknown at session start).
- Limitation: one year of data, so August never appears in training.

## Regression findings
- Baseline (predict the average) gives R2 ~ 0.00, MAE ~ 10.9.
- All real models land within a narrow band (R2 ~ 0.31 - 0.35).
- Best model: LightGBM , R2 = 0.351, MAE = 8.159 (about 25% better than baseline).
- The gaps between models are small, so I do not claim one is decisively better.
- Predictions regress toward the mean: low demand is over-predicted, high demand under-predicted.
- Leakage test: adding Duration lifts R2 to ~0.79, but Duration is unknown at session start,
  so it is excluded from the real model.

  ## Forecasting findings
- Daily demand averages about 3,480 kWh, with a clear weekly pattern (Thursday highest, Sunday lowest).
- Evaluated with a rolling backtest: 12 weekly origins, 7-day horizon.
- All methods land within a narrow band (MAE about 490-580 kWh/day, about 14-17% of average demand).
- I chose a seasonal-average forecast for the weekday shape.
- Limitations: only one year of data (no annual seasonality can be learned), and an apparent outage
  on 5 September 2022 (42 sessions vs about 220 normal) was kept in the data.