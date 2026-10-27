"""Loading, filtering and summarising the cleaned charging sessions."""
import pandas as pd

from src.config import CLEAN_DATA_PATH

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_sessions():
    """Read the cleaned sessions and add the columns the dashboard needs."""
    sessions = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["StartDatetime", "EndDatetime"])
    sessions["Hour"] = sessions["StartDatetime"].dt.hour
    sessions["DayOfWeek"] = sessions["StartDatetime"].dt.dayofweek
    sessions["Date"] = sessions["StartDatetime"].dt.normalize()
    return sessions


def filter_sessions(sessions, start_date, end_date, locations, charger_types, companies):
    """Keep sessions inside the date range. An empty list means 'do not filter on this'."""
    mask = sessions["Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    if locations:
        mask &= sessions["Location"].isin(locations)
    if charger_types:
        mask &= sessions["ChargerTypeLabel"].isin(charger_types)
    if companies:
        mask &= sessions["ChargerCompanyLabel"].isin(companies)
    return sessions[mask]


def compute_kpis(sessions):
    """Headline numbers for the Overview tab."""
    return {
        "sessions": len(sessions),
        "total_demand": sessions["Demand"].sum(),
        "mean_demand": sessions["Demand"].mean() if len(sessions) else 0.0,
        "chargers": sessions["ChargerID"].nunique(),
        "known_users": sessions.loc[sessions["IsKnownUser"], "UserID"].nunique(),
    }


def demand_by(sessions, column):
    """Number of sessions, average and total demand for each value of one column."""
    summary = sessions.groupby(column)["Demand"].agg(
        Sessions="size", AvgDemand="mean", TotalDemand="sum")
    return summary.round(2).reset_index().sort_values("AvgDemand", ascending=False)


def hourly_profile(sessions):
    """Sessions and average demand for each start hour."""
    profile = sessions.groupby("Hour")["Demand"].agg(Sessions="size", AvgDemand="mean")
    return profile.reindex(range(24)).fillna(0).rename_axis("Hour").reset_index()


def weekday_hour_table(sessions):
    """Number of sessions for every weekday and hour (rows = weekday, columns = hour)."""
    table = sessions.pivot_table(
        index="DayOfWeek", columns="Hour", values="Demand", aggfunc="size", fill_value=0)
    table = table.reindex(index=range(7), columns=range(24), fill_value=0)
    table.index = WEEKDAYS
    return table


def daily_series(sessions):
    """One row per calendar day: sessions, total demand and a 7-day average of sessions."""
    daily = sessions.set_index("StartDatetime").resample("D").agg(
        Sessions=("Demand", "size"), TotalDemand=("Demand", "sum"))
    daily["Sessions_7d"] = daily["Sessions"].rolling(7).mean()
    return daily.reset_index()