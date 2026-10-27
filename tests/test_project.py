"""Simple checks that the dashboard code builds the same features as training.

Run from the project root:   python tests/test_project.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import data, explain, predict  # noqa: E402
from src.config import TEST_DATA_PATH  # noqa: E402


def test_feature_row_matches_training_rows():
    """Rebuild 300 real test sessions from their raw inputs and compare with the stored features."""
    settings = predict.load_settings()
    features = settings["features"]
    test = pd.read_csv(TEST_DATA_PATH, parse_dates=["StartDatetime"]).sample(300, random_state=1)

    for _, session in test.iterrows():
        rebuilt = predict.build_feature_row(
            settings, session["Location"], session["ChargerType"], session["ChargerCompany"],
            session["StartDatetime"].to_pydatetime(),
            int(session["ChargerPrevSessions"]), session["ChargerPrevAvgDemand"],
            int(session["UserPrevSessions"]), session["UserPrevAvgDemand"])
        expected = session[features].astype(float).to_frame().T.reset_index(drop=True)
        pd.testing.assert_frame_equal(rebuilt.astype(float), expected, check_dtype=False)


def test_time_of_day_boundaries():
    assert predict.time_of_day(0) == "Night"
    assert predict.time_of_day(5) == "Night"
    assert predict.time_of_day(6) == "Morning"
    assert predict.time_of_day(16) == "Afternoon"
    assert predict.time_of_day(17) == "Evening"
    assert predict.time_of_day(21) == "LateEvening"
    assert predict.time_of_day(23) == "LateEvening"


def test_unknown_location_is_rejected():
    from datetime import datetime
    settings = predict.load_settings()
    try:
        predict.build_feature_row(settings, "moon base", 0, 0, datetime(2022, 10, 3, 18, 0))
    except ValueError:
        return
    raise AssertionError("an unknown location should raise ValueError")


def test_no_history_uses_training_mean():
    from datetime import datetime
    settings = predict.load_settings()
    row = predict.build_feature_row(settings, predict.location_options(settings)[0], 0, 0,
                                    datetime(2022, 10, 3, 18, 0))
    assert row.loc[0, "HasChargerHistory"] == 0
    assert row.loc[0, "ChargerPrevAvgDemand"] == settings["history_fill_value"]
    assert list(row.columns) == settings["features"]


def test_filters_and_summaries():
    sessions = data.load_sessions()
    everything = data.filter_sessions(sessions, sessions["Date"].min(), sessions["Date"].max(), [], [], [])
    assert len(everything) == len(sessions)
    one_type = data.filter_sessions(sessions, sessions["Date"].min(), sessions["Date"].max(),
                                    [], ["Type 1"], [])
    assert 0 < len(one_type) < len(sessions)
    assert data.compute_kpis(one_type)["sessions"] == len(one_type)
    assert data.weekday_hour_table(sessions).shape == (7, 24)
    assert len(data.hourly_profile(one_type)) == 24
    assert set(data.demand_by(sessions, "Location").columns) >= {"Sessions", "AvgDemand"}
    assert data.compute_kpis(sessions.iloc[0:0])["mean_demand"] == 0.0


def test_shap_reshaping_handles_both_layouts():
    """A fake explainer checks our reshaping, without needing SHAP installed."""
    import numpy as np

    class FakeExplanation:
        def __init__(self, values, base_values):
            self.values, self.base_values = values, base_values

    row = pd.DataFrame([{"a": 1.0, "b": 2.0, "c": 3.0}])
    two_dimensional = lambda r: FakeExplanation(np.array([[0.5, -2.0, 1.0]]), np.array([10.0]))
    three_dimensional = lambda r: FakeExplanation(
        np.array([[[0, 0.5], [0, -2.0], [0, 1.0]]]), np.array([[0.0, 10.0]]))

    for fake in (two_dimensional, three_dimensional):
        table, base = explain.shap_contributions(fake, row)
        assert base == 10.0
        assert list(table["Feature"]) == ["b", "c", "a"]      # largest effect first
        assert abs(table["Contribution"].sum() - (-0.5)) < 1e-9


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)
    print(f"All {len(tests)} tests passed")