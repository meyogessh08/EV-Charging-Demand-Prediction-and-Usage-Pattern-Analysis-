"""Turn a user's inputs into the exact feature row the models were trained on."""
import json

import joblib
import pandas as pd

from src.config import CLASSIFIER_PATH, FEATURE_SETTINGS_PATH, REGRESSOR_PATH

# (first hour, last hour, label). Must match the bins used in notebook 04.
TIME_OF_DAY = [(0, 5, "Night"), (6, 11, "Morning"), (12, 16, "Afternoon"),
               (17, 20, "Evening"), (21, 23, "LateEvening")]


def load_settings():
    """Feature list, High Demand threshold and the fill value saved in notebook 04."""
    with open(FEATURE_SETTINGS_PATH) as file:
        return json.load(file)


def load_models():
    """The pickled regression and classification models."""
    return joblib.load(REGRESSOR_PATH), joblib.load(CLASSIFIER_PATH)


def location_options(settings):
    """Readable location names, taken from the Loc_ columns the models know about."""
    names = [name[4:].replace("_", " ") for name in settings["features"] if name.startswith("Loc_")]
    return sorted(names)


def time_of_day(hour):
    """Name of the part of the day for an hour (0-23)."""
    for first, last, label in TIME_OF_DAY:
        if first <= hour <= last:
            return label
    raise ValueError(f"Hour must be between 0 and 23, got {hour}")


def build_feature_row(settings, location, charger_type, charger_company, start,
                      charger_prev_sessions=0, charger_prev_avg_demand=None,
                      user_prev_sessions=0, user_prev_avg_demand=None):
    """Return a one-row DataFrame in exactly the column order used for training.

    Missing history is handled like in training: the 'Has...History' flag is 0
    and the average is replaced by the saved training mean.
    """
    features = settings["features"]
    fill_value = settings["history_fill_value"]

    if charger_prev_sessions > 0 and charger_prev_avg_demand is None:
        raise ValueError("Give the charger's average demand when it has earlier sessions")
    if user_prev_sessions > 0 and user_prev_avg_demand is None:
        raise ValueError("Give the user's average demand when the user has earlier sessions")

    row = dict.fromkeys(features, 0)
    row.update({
        "ChargerCompany": int(charger_company),
        "ChargerType": int(charger_type),
        "Hour": start.hour,
        "Day": start.day,
        "Month": start.month,
        "DayOfWeek": start.weekday(),
        "IsWeekend": int(start.weekday() >= 5),
        "ChargerPrevSessions": charger_prev_sessions,
        "ChargerPrevAvgDemand": charger_prev_avg_demand if charger_prev_sessions > 0 else fill_value,
        "HasChargerHistory": int(charger_prev_sessions > 0),
        "UserPrevSessions": user_prev_sessions,
        "UserPrevAvgDemand": user_prev_avg_demand if user_prev_sessions > 0 else fill_value,
        "HasUserHistory": int(user_prev_sessions > 0),
    })

    for column in ("Loc_" + location.replace(" ", "_"), "TOD_" + time_of_day(start.hour)):
        if column not in row:
            raise ValueError(f"The model does not know this category: {column}")
        row[column] = 1

    return pd.DataFrame([row], columns=features)


def predict_demand(regressor, row):
    """Predicted demand of one session."""
    return float(regressor.predict(row)[0])


def predict_high_probability(classifier, row):
    """Probability that the session is High Demand."""
    return float(classifier.predict_proba(row)[0, 1])