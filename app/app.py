"""EV Charging Demand dashboard.

Run from the project root:   streamlit run app/app.py
"""

import sys
from datetime import datetime, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # lets us write "from src import ..." when Streamlit starts this file

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import data, explain, predict  # noqa: E402
from src.config import ACCENT, FIGURE_DIR, MODEL_DIR, SECONDARY, TRAIN_DATA_PATH, WARNING  # noqa: E402

st.set_page_config(page_title="EV Charging Demand", layout="wide")


# ---------- Cached loaders (each runs once, then the result is reused) ----------
@st.cache_data
def get_sessions():
    return data.load_sessions()


@st.cache_data
def get_settings():
    return predict.load_settings()


@st.cache_resource
def get_models():
    return predict.load_models()


@st.cache_resource
def get_shap_explainers():
    regressor, classifier = get_models()
    return explain.get_shap_explainer(regressor), explain.get_shap_explainer(classifier)


@st.cache_resource
def get_lime_explainers():
    features = get_settings()["features"]
    background = pd.read_csv(TRAIN_DATA_PATH, usecols=features).sample(5000, random_state=42)
    return (
        explain.build_lime_explainer(background, "regression"),
        explain.build_lime_explainer(background, "classification"),
    )


@st.cache_data
def read_table(file_name, **options):
    """Read one of the result tables saved by the notebooks (or None if it does not exist yet)."""
    path = MODEL_DIR / file_name
    return pd.read_csv(path, **options) if path.exists() else None


# ---------- Small chart helpers ----------
def effect_chart(labels, values, x_label):
    """Horizontal bars: green pushes the prediction up, red pushes it down."""
    colours = [ACCENT if value >= 0 else WARNING for value in values]
    figure = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colours))
    figure.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=x_label,
        template="plotly_white",
    )
    return figure


def contribution_labels(table):
    return [f"{name} = {value:g}" for name, value in zip(table["Feature"], table["Value"])]


def show_image(file_name, caption):
    path = FIGURE_DIR / file_name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"{file_name} not found. Run the notebooks that create it.")


# ---------- Page header and sidebar filters ----------
sessions = get_sessions()
settings = get_settings()

st.title("EV Charging Demand: Usage Patterns and Prediction")
st.caption(
    "Explore charging behaviour, forecast demand, and predict a single session with explanations."
)

first_day, last_day = sessions["Date"].min().date(), sessions["Date"].max().date()
st.sidebar.header("Filters")
st.sidebar.caption("Filters apply to the Overview and Demand Patterns tabs.")
date_range = st.sidebar.date_input(
    "Date range", value=(first_day, last_day), min_value=first_day, max_value=last_day
)
if len(date_range) != 2:
    st.sidebar.info("Pick both a start and an end date.")
    st.stop()
start_date, end_date = date_range
chosen_locations = st.sidebar.multiselect(
    "Location (empty = all)", sorted(sessions["Location"].unique())
)
chosen_types = st.sidebar.multiselect(
    "Charger type (empty = all)", sorted(sessions["ChargerTypeLabel"].unique())
)
chosen_companies = st.sidebar.multiselect(
    "Charger company (empty = all)", sorted(sessions["ChargerCompanyLabel"].unique())
)

filtered = data.filter_sessions(
    sessions, start_date, end_date, chosen_locations, chosen_types, chosen_companies
)
st.sidebar.write(f"{len(filtered):,} of {len(sessions):,} sessions selected")

overview_tab, patterns_tab, forecast_tab, predict_tab, models_tab = st.tabs(
    ["Overview", "Demand Patterns", "Forecast", "Predict a Session", "Model Insights"]
)

# ---------- 1. Overview ----------
with overview_tab:
    if filtered.empty:
        st.warning("No sessions match the current filters.")
    else:
        kpis = data.compute_kpis(filtered)
        cards = [
            ("Sessions", f"{kpis['sessions']:,}"),
            ("Total demand (kWh)", f"{kpis['total_demand']:,.0f}"),
            ("Average per session (kWh)", f"{kpis['mean_demand']:.1f}"),
            ("Chargers used", f"{kpis['chargers']:,}"),
            ("Known users", f"{kpis['known_users']:,}"),
        ]
        for column, (label, value) in zip(st.columns(5), cards):
            with column.container(border=True):
                st.caption(label)
                st.subheader(value)

        daily = data.daily_series(filtered).rename(
            columns={"Sessions": "Daily sessions", "Sessions_7d": "7-day average"}
        )
        st.subheader("Sessions per day")
        trend = px.line(
            daily,
            x="StartDatetime",
            y=["Daily sessions", "7-day average"],
            color_discrete_sequence=["#B7C9C2", ACCENT],
        )
        trend.update_layout(
            height=340,
            template="plotly_white",
            xaxis_title="",
            yaxis_title="Sessions",
            legend_title_text="",
        )
        st.plotly_chart(trend)

        hourly = data.hourly_profile(filtered)
        left, right = st.columns(2)
        with left:
            st.subheader("Sessions by start hour")
            figure = px.bar(hourly, x="Hour", y="Sessions", color_discrete_sequence=[ACCENT])
            figure.update_layout(height=320, template="plotly_white")
            st.plotly_chart(figure)
        with right:
            st.subheader("Average demand by start hour (kWh)")
            figure = px.line(
                hourly, x="Hour", y="AvgDemand", markers=True, color_discrete_sequence=[WARNING]
            )
            figure.update_layout(height=320, template="plotly_white")
            st.plotly_chart(figure)

# ---------- 2. Demand patterns ----------
with patterns_tab:
    if filtered.empty:
        st.warning("No sessions match the current filters.")
    else:
        st.subheader("Demand by location")
        by_location = data.demand_by(filtered, "Location")
        figure = px.bar(
            by_location,
            x="AvgDemand",
            y="Location",
            orientation="h",
            hover_data=["Sessions"],
            color_discrete_sequence=[SECONDARY],
        )
        figure.update_layout(
            height=420,
            template="plotly_white",
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Average demand per session (kWh)",
        )
        st.plotly_chart(figure)
        st.caption(
            "Check the Sessions column: locations with few sessions give less reliable averages."
        )
        st.dataframe(by_location, hide_index=True)

        left, right = st.columns(2)
        for column, field, title in [
            (left, "ChargerTypeLabel", "Demand by charger type"),
            (right, "ChargerCompanyLabel", "Demand by charger company"),
        ]:
            with column:
                st.subheader(title)
                summary = data.demand_by(filtered, field)
                figure = px.bar(
                    summary,
                    x=field,
                    y="AvgDemand",
                    hover_data=["Sessions"],
                    color_discrete_sequence=[ACCENT],
                )
                figure.update_layout(
                    height=320,
                    template="plotly_white",
                    xaxis_title="",
                    yaxis_title="Average demand (kWh)",
                )
                st.plotly_chart(figure)

        st.subheader("When are chargers busy? (sessions by weekday and hour)")
        heatmap = px.imshow(
            data.weekday_hour_table(filtered),
            aspect="auto",
            color_continuous_scale="Greens",
            labels=dict(x="Start hour", y="", color="Sessions"),
        )
        heatmap.update_layout(height=360)
        st.plotly_chart(heatmap)

# ---------- 3. Forecast ----------
with forecast_tab:
    st.subheader("Total daily demand: next 14 days")
    st.caption("This forecast covers the whole network and ignores the sidebar filters.")
    forecast = read_table("forecast_next_14_days.csv", parse_dates=["Date"])
    if forecast is None:
        st.info("Run notebook 07 to create the forecast.")
    else:
        history = data.daily_series(sessions)
        history = history[history["StartDatetime"] >= "2021-10-01"].tail(60)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=history["StartDatetime"],
                y=history["TotalDemand"],
                name="Actual (last 60 days)",
                line=dict(color=SECONDARY),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=forecast["Date"],
                y=forecast["ForecastDemand_kWh"],
                name="Forecast",
                line=dict(color=WARNING),
                mode="lines+markers",
            )
        )
        figure.update_layout(height=380, template="plotly_white", yaxis_title="Demand (kWh)")
        st.plotly_chart(figure)

        backtest = read_table("forecast_backtest.csv", index_col=0)
        if backtest is not None:
            st.subheader("How accurate are the forecasting methods?")
            st.caption(
                "Average error over 12 weekly forecasts of the following 7 days (lower is better)."
            )
            st.dataframe(backtest.round(1))

# ---------- 4. Predict a session ----------
with predict_tab:
    st.subheader("Predict demand for a new session")
    st.write(
        "Describe the session and what is already known before it starts. "
        "The models were trained without the session's duration, because that is unknown in advance."
    )

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            location = st.selectbox("Location", predict.location_options(settings))
            charger_type = st.selectbox(
                "Charger type", [0, 1], format_func=lambda value: f"Type {value}"
            )
            charger_company = st.selectbox(
                "Charger company", [0, 1], format_func=lambda value: f"Company {value}"
            )
            start_day = st.date_input("Start date", value=pd.Timestamp("2022-10-03").date())
            start_clock = st.time_input("Start time", value=time(18, 0))
        with right:
            st.caption("History known before the session starts")
            charger_sessions = st.number_input(
                "Charger: earlier sessions", min_value=0, value=50, step=10
            )
            charger_average = st.number_input(
                "Charger: average demand of earlier sessions (kWh)",
                min_value=0.0,
                value=17.0,
                step=1.0,
            )
            user_sessions = st.number_input(
                "User: earlier sessions (0 = unknown user)", min_value=0, value=0, step=1
            )
            user_average = st.number_input(
                "User: average demand of earlier sessions (kWh)",
                min_value=0.0,
                value=17.0,
                step=1.0,
            )
            with_lime = st.checkbox("Also show a LIME explanation (slower)")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        start = datetime.combine(start_day, start_clock)
        row = predict.build_feature_row(
            settings,
            location,
            charger_type,
            charger_company,
            start,
            charger_sessions,
            charger_average,
            user_sessions,
            user_average,
        )
        regressor, classifier = get_models()
        predicted_demand = predict.predict_demand(regressor, row)
        probability_high = predict.predict_high_probability(classifier, row)
        threshold = settings["high_demand_threshold"]

        first, second, third = st.columns(3)
        with first.container(border=True):
            st.caption("Predicted demand (kWh)")
            st.subheader(f"{predicted_demand:.1f}")
        with second.container(border=True):
            st.caption("Probability of High Demand")
            st.subheader(f"{probability_high:.0%}")
        with third.container(border=True):
            st.caption(f"Class (High means at least {threshold:.1f} kWh)")
            st.subheader("High Demand" if probability_high >= 0.5 else "Normal Demand")

        regression_results = read_table("regression_results.csv", index_col=0)
        if regression_results is not None:
            st.caption(
                f"On unseen data the regression model's average error was about "
                f"{regression_results['MAE'].min():.1f} kWh, so treat this as an estimate, not a promise."
            )

        st.subheader("Why this prediction? (SHAP)")
        try:
            regression_explainer, classification_explainer = get_shap_explainers()
            regression_table, regression_base = explain.shap_contributions(
                regression_explainer, row
            )
            classification_table, classification_base = explain.shap_contributions(
                classification_explainer, row
            )
            left, right = st.columns(2)
            with left:
                st.write(
                    f"Demand model (starts from the average prediction of {regression_base:.1f} kWh)"
                )
                top = regression_table.head(10).iloc[::-1]
                st.plotly_chart(
                    effect_chart(
                        contribution_labels(top), top["Contribution"], "Effect on demand (kWh)"
                    )
                )
            with right:
                st.write("High Demand model (effects in log-odds, not percentages)")
                top = classification_table.head(10).iloc[::-1]
                st.plotly_chart(
                    effect_chart(
                        contribution_labels(top), top["Contribution"], "Effect (log-odds)"
                    )
                )
            st.caption(
                "Green bars push the prediction up, red bars push it down. "
                "These show how the model uses its inputs, not what causes demand."
            )
        except Exception as error:
            st.warning(f"SHAP explanation unavailable: {error}")

        if with_lime:
            st.subheader("A second opinion (LIME)")
            try:
                features = settings["features"]
                lime_regression, lime_classification = get_lime_explainers()
                regressor, classifier = get_models()
                lime_demand = explain.lime_contributions(
                    lime_regression,
                    lambda values: regressor.predict(pd.DataFrame(values, columns=features)),
                    row,
                )
                lime_class = explain.lime_contributions(
                    lime_classification,
                    lambda values: classifier.predict_proba(
                        pd.DataFrame(values, columns=features)
                    ),
                    row,
                )
                left, right = st.columns(2)
                with left:
                    st.write("Demand model")
                    flipped = lime_demand.iloc[::-1]
                    st.plotly_chart(
                        effect_chart(flipped["Rule"], flipped["Effect"], "Effect on demand (kWh)")
                    )
                with right:
                    st.write("High Demand model")
                    flipped = lime_class.iloc[::-1]
                    st.plotly_chart(
                        effect_chart(flipped["Rule"], flipped["Effect"], "Effect on probability")
                    )
                st.caption(
                    "LIME is a local approximation built from random samples, so small effects can vary "
                    "from run to run. Trust the largest bars."
                )
            except Exception as error:
                st.warning(f"LIME explanation unavailable: {error}")

# ---------- 5. Model insights ----------
with models_tab:
    st.subheader("Regression: predicting demand (kWh)")
    regression_table = read_table("regression_results.csv", index_col=0)
    if regression_table is not None:
        st.dataframe(regression_table.round(3))
    st.subheader("Classification: High vs Normal Demand")
    classification_table = read_table("classification_results.csv", index_col=0)
    if classification_table is not None:
        st.dataframe(classification_table.round(3))
    imbalance_table = read_table("imbalance_experiment.csv", index_col=0)
    if imbalance_table is not None:
        st.subheader("Class-imbalance experiment (stricter 'peak demand' target)")
        st.caption(
            "Accuracy is misleading here: always guessing 'not peak' is already about 89% accurate."
        )
        st.dataframe(imbalance_table.round(3))

    st.subheader("Which factors matter most?")
    importance = read_table("feature_importance.csv", index_col=0)
    if importance is not None:
        top = importance.sort_values("permutation_mae_rise", ascending=False).head(12).iloc[::-1]
        figure = px.bar(
            top,
            x="permutation_mae_rise",
            y=top.index,
            orientation="h",
            color_discrete_sequence=[SECONDARY],
        )
        figure.update_layout(
            height=420,
            template="plotly_white",
            yaxis_title="",
            xaxis_title="Increase in prediction error when the feature is shuffled (kWh)",
        )
        st.plotly_chart(figure)

    st.subheader("SHAP summaries")
    left, right = st.columns(2)
    with left:
        show_image("19_shap_regression_beeswarm.png", "Regression model")
    with right:
        show_image("22_shap_classification_beeswarm.png", "High Demand classifier")
