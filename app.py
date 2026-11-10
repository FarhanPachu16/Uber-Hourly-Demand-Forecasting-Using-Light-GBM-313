import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

import forecast_utils as fu

st.set_page_config(page_title="NYC Uber demand forecast", layout="wide")


@st.cache_data
def get_data(horizon_hours):
    return fu.load_forecasts(horizon_hours)


st.title("Hourly Uber demand forecasts for NYC taxi zones")
st.caption(
    "August 2026 test-set forecasts from a LightGBM model trained on January-June 2026 NYC TLC trip data. "
    "These are precomputed from the saved test run, not live predictions."
)

# ---------------- sidebar ----------------
horizon_label = st.sidebar.radio("Forecast horizon", list(fu.HORIZONS))
df = get_data(fu.HORIZONS[horizon_label])
first_day, last_day = df["hour"].min().date(), df["hour"].max().date()


def clamp(day):
    return min(max(day, first_day), last_day)


zones = fu.zone_options(df)
zone_label = st.sidebar.selectbox("Zone (busiest first)", zones["zone_label"], index=0)
zone_id = int(zones.loc[zones["zone_label"] == zone_label, "zone_id"].iloc[0])
start = st.sidebar.date_input("Start date", value=clamp(dt.date(2026, 8, 10)),
                              min_value=first_day, max_value=last_day)
days = st.sidebar.slider("Days to show", 1, 14, 7)
show_band = st.sidebar.checkbox("Show 80% prediction interval", value=True)

tab_zone, tab_city, tab_about = st.tabs(["Zone forecast", "Where will demand be?", "Results and method"])

# ---------------- tab 1: one zone over time ----------------
with tab_zone:
    w = fu.zone_window(df, zone_id, start, days)
    if w.empty:
        st.warning("No data for this zone and date range.")
    else:
        s = fu.summary(w)
        c1, c2, c3 = st.columns(3)
        c1.metric("Model error (WAPE)", f"{s['model_wape']:.1f}%",
                  delta=f"{s['model_wape'] - s['baseline_wape']:+.1f} pts vs same hour last week",
                  delta_color="inverse")
        c2.metric("Same hour last week (WAPE)", f"{s['baseline_wape']:.1f}%")
        c3.metric("80% interval coverage", f"{s['coverage']:.0f}%")

        names = {"actual": "Actual", "forecast": "Model forecast",
                 "last_week_baseline": "Same hour last week"}
        long = w.melt(id_vars="hour", value_vars=list(names), var_name="series", value_name="trips")
        long["series"] = long["series"].map(names)

        lines = alt.Chart(long).mark_line().encode(
            x=alt.X("hour:T", title=None),
            y=alt.Y("trips:Q", title="Uber trips per hour"),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=list(names.values()),
                                            range=["#2b2b2b", "#d1495b", "#9aa5b1"])),
        )
        band = alt.Chart(w).mark_area(opacity=0.18, color="#d1495b").encode(
            x="hour:T", y="lo80:Q", y2="hi80:Q")
        chart = (band + lines) if show_band else lines
        st.altair_chart(chart.properties(width="container", height=380))

        with st.expander("Show the numbers"):
            table = w[["hour", "actual", "forecast", "lo80", "hi80", "last_week_baseline"]].copy()
            number_cols = table.columns.difference(["hour"])
            table[number_cols] = table[number_cols].round(1)
            table.columns = ["Hour", "Actual", "Forecast", "Interval low", "Interval high", "Same hour last week"]
            st.dataframe(table, hide_index=True)

# ---------------- tab 2: all zones at one hour ----------------
with tab_city:
    st.write("Pick an hour to see which zones the model expected to be busiest, and what actually happened. "
             "This is the 'where should drivers be?' view.")
    col_a, col_b, col_c = st.columns(3)
    day = col_a.date_input("Date", value=clamp(dt.date(2026, 8, 14)),
                           min_value=first_day, max_value=last_day, key="snapshot_date")
    hour_of_day = col_b.slider("Hour of day", 0, 23, 18)
    top_n = col_c.slider("Number of zones", 5, 25, 15)

    when = pd.Timestamp(day) + pd.Timedelta(hours=hour_of_day)
    snap = fu.hour_snapshot(df, when)
    if snap.empty:
        st.warning("No data for that hour.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("City-wide trips: actual", f"{snap['actual'].sum():,.0f}")
        m2.metric("City-wide trips: forecast", f"{snap['forecast'].sum():,.0f}")
        m3.metric(f"Demand held by the forecast's top {top_n} zones", f"{fu.topn_capture(snap, top_n):.0f}%",
                  help="Compared with the demand held by the truly busiest zones that hour (100% = perfect).")

        top = snap.nlargest(top_n, "forecast")
        order = list(top["zone_label"])
        bars = top.melt(id_vars="zone_label", value_vars=["forecast", "actual"],
                        var_name="series", value_name="trips")
        bars["series"] = bars["series"].map({"forecast": "Forecast", "actual": "Actual"})
        chart = alt.Chart(bars).mark_bar().encode(
            y=alt.Y("zone_label:N", sort=order, title=None),
            x=alt.X("trips:Q", title="Uber trips in the hour"),
            yOffset="series:N",
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=["Forecast", "Actual"], range=["#d1495b", "#2b2b2b"])),
        )
        st.altair_chart(chart.properties(width="container", height=max(300, 26 * top_n)))

# ---------------- tab 3: results and method ----------------
with tab_about:
    st.subheader("August test results (whole month, all 261 zones)")
    st.dataframe(fu.results_overview(), hide_index=True)
    st.markdown(
        """
**Metric.** WAPE = total absolute error / total actual trips. It is used instead of MAPE because many
zone-hours have zero or very few trips.

**Setup.** Training: January to June 2026. Validation (model settings, interval calibration): July.
Test: August, evaluated once. Time-based split, no shuffling, and every history feature is at least
as old as the forecast horizon, so nothing from the future leaks in.

**Model.** One global LightGBM model (Poisson loss) for all zones, using calendar and holiday features,
lagged trips (1, 2, 3, 24, 48 and 168 hours back where allowed by the horizon), rolling averages and the
previous wait time. Prediction intervals come from 10% and 90% quantile models, widened by a conformal
adjustment calibrated on July.

**Limitations.**
- Only completed Uber trips are in the data, so this is fulfilled demand, not total demand.
- Eight months of one year: the model has never seen July 4, Labor Day, Thanksgiving or Christmas.
- No flight or event data, which matters most at the airports.
- The test month has no major holiday.
- Observed weather was tested and did not help, so it is not used.

Data: NYC Taxi and Limousine Commission trip records. Not affiliated with Uber.
"""
    )
