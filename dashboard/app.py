"""M10: public Streamlit dashboard (blueprint 14). Reads PostgreSQL directly until M12 switches it to the API.

Run locally from the project root (DATABASE_URL from .env):
    uv run streamlit run dashboard/app.py

On Streamlit Community Cloud, DATABASE_URL is set as a secret; top-level secrets become environment
variables, which `get_connection` reads. The connection is read-only: the dashboard never writes.

Four tabs: Scorecard (live scores, provisional and settled, plus rolling performance), Forecasts (price
history + the latest 7-day forecasts per series), Data health, Incidents (docs/incidents.md).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Community Cloud installs dashboard/requirements.txt but not this project itself, so `mandi` is imported
# straight from src/. Locally (`uv run`) the package is installed and this changes nothing.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from mandi.evaluate.live_report import (
    MIN_SCORED_FORECASTS,
    ROLLING_WINDOW_DAYS,
    load_latest_forecasts,
    load_live_scores,
    price_history,
    rolling_scores,
    scorecard,
)
from mandi.evaluate.scoring import SETTLING_DAYS, scoring_counts
from mandi.ingest.snapshot import IST
from mandi.models.daily_forecast import prices_by_series, series_info_table
from mandi.monitor.health import load_data_health
from mandi.store.db import get_connection

CACHE_SECONDS = 600  # the data changes once a day; re-reading at most every 10 minutes spares Neon compute
HISTORY_DAYS = 90


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Reading the database (the first load can take a while)...")
def load_dashboard_data() -> dict:
    """Everything the page shows, read in one read-only connection."""
    load_dotenv(PROJECT_ROOT / ".env")
    now = datetime.now(IST)
    with get_connection() as conn:
        conn.read_only = True
        return {
            "read_at": now,
            "scores": load_live_scores(conn, now),
            "counts": pd.DataFrame(scoring_counts(conn, now.date())),
            "series": series_info_table(conn),
            "prices": prices_by_series(conn, now),
            "forecasts": load_latest_forecasts(conn),
            "health": load_data_health(conn),
        }


def show_scorecard_table(title: str, scores: pd.DataFrame, empty_message: str) -> None:
    st.subheader(title)
    table = scorecard(scores, ["crop", "horizon_days"])
    if table.empty:
        st.info(empty_message)
        return
    st.dataframe(table, hide_index=True, width="stretch")


def show_scorecard_tab(data: dict) -> None:
    scores = data["scores"]
    st.markdown(
        f"""
Live forecasts of the modal price (₹/quintal), scored against the price the source later reports.
Models: **naive** (last price), **seasonal naive** (price 7 days before the target) and **LightGBM**.

- **Like-for-like:** each table uses only the forecasts that *every* model scored, so **n** is the same
  for all models in a row. Rows where LightGBM loses are shown like any other.
- **Provisional** scores use the latest reported value, which the source may still edit. A score becomes
  **settled** once its target date is more than {SETTLING_DAYS} days old; each forecast counts once, in one
  table or the other.
- **n below {MIN_SCORED_FORECASTS}** is marked *insufficient data*: too few to say which model is better.
- **MASE** below 1 means a smaller error than the seasonal-naive forecast had on that series' own history
  before the forecast was made.
- In the M08 backtest (on the latest history, not live) LightGBM lost to naive; see
  `docs/backtest_report.md` in the repository.
"""
    )
    show_scorecard_table(
        "Settled",
        scores[scores["status"] == "settled"],
        f"No settled scores yet. A target date settles {SETTLING_DAYS + 1} days after it, once its price "
        "can no longer be edited.",
    )
    show_scorecard_table(
        "Provisional",
        scores[scores["status"] == "provisional"],
        "No provisional scores yet. A forecast is scored once a price is reported for its target date.",
    )

    st.subheader("Forecasts per model")
    st.caption(
        "Scored = has a score; unreported so far = the target date passed without a reported price; "
        "unavailable = the model could not make that forecast (e.g. seasonal naive when the day 7 days "
        "earlier was unreported)."
    )
    st.dataframe(data["counts"], hide_index=True, width="stretch")

    st.subheader(f"Performance over time ({ROLLING_WINDOW_DAYS}-day window)")
    rolling = rolling_scores(scores)
    if rolling.empty:
        st.info("No scores yet.")
        return
    metric = st.radio("Metric", ["MASE", "MAE"], horizontal=True)
    st.caption(
        f"Each point pools all series and horizons whose target date falls in the {ROLLING_WINDOW_DAYS} days "
        f"ending that date (settled score if there is one, else the latest provisional). Windows with fewer "
        f"than {MIN_SCORED_FORECASTS} scores per model are not drawn; hover a point for its n."
    )
    chart = (
        alt.Chart(rolling.dropna(subset=[metric]))
        .mark_line(point=True)
        .encode(
            x=alt.X("end_date:T", title="window end (target date)"),
            y=alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(zero=False)),
            color=alt.Color("model_version:N", title="model"),
            tooltip=["end_date:T", "model_version:N", "n:Q", alt.Tooltip(f"{metric}:Q", format=".3f")],
        )
    )
    if rolling[metric].notna().any():
        st.altair_chart(chart, width="stretch")
    else:
        st.info(f"No window has {MIN_SCORED_FORECASTS} scores per model yet.")
    newest_first = rolling.sort_values(["end_date", "model_version"], ascending=[False, True])
    st.dataframe(newest_first.round({"MAE": 1, "MASE": 3}), hide_index=True)


def show_forecasts_tab(data: dict) -> None:
    series = data["series"]
    labels = {}
    for row in series.itertuples(index=False):
        labels[f"{row.crop}, {row.market}"] = row.series_id
    choice = st.selectbox("Series", list(labels))
    series_id = labels[choice]

    forecasts = data["forecasts"]
    forecasts = forecasts[forecasts["series_id"] == series_id]
    first_day = data["read_at"].date() - timedelta(days=HISTORY_DAYS)
    history = price_history(data["prices"].get(series_id, {}), first_day)
    history["line"] = "reported price"

    forecast_points = forecasts[forecasts["point_forecast"].notna()]
    forecast_points = pd.DataFrame(
        {
            "date": forecast_points["target_date"],
            "price": forecast_points["point_forecast"],
            "line": forecast_points["model_version"],
        }
    )
    # Reported prices are dots, not a line: a line would draw prices on unreported days that never existed.
    reported = alt.Chart(history).mark_point(filled=True)
    predicted = alt.Chart(forecast_points).mark_line(point=True)
    encoding = {
        "x": alt.X("date:T", title="date"),
        "y": alt.Y("price:Q", title="modal price (₹/quintal)", scale=alt.Scale(zero=False)),
        "color": alt.Color("line:N", title=""),
        "tooltip": ["date:T", "line:N", alt.Tooltip("price:Q", format=",.0f")],
    }
    st.altair_chart(reported.encode(**encoding) + predicted.encode(**encoding), width="stretch")

    if forecasts.empty:
        st.info("No forecasts for this series yet.")
        return
    origin = forecasts["origin_date"].iloc[0]
    input_as_of = forecasts["input_as_of_at"].max().astimezone(IST)
    st.caption(
        f"Last {HISTORY_DAYS} days of reported prices; missing days were not reported. Forecasts made on "
        f"{origin} with prices seen by {input_as_of:%Y-%m-%d %H:%M} IST. Point forecasts only; "
        "intervals are not built yet."
    )
    st.dataframe(
        forecasts[["model_version", "horizon_days", "target_date", "point_forecast", "unavailable_reason"]],
        hide_index=True,
        width="stretch",
    )


def show_health_tab(data: dict) -> None:
    health = data["health"]
    st.subheader("Pipeline runs")
    st.caption("The latest run of each job, and when that job last succeeded. Times in UTC.")
    st.dataframe(health.last_runs, hide_index=True, width="stretch")

    st.subheader("Days per series")
    st.caption(
        "All history. Unreported = no row from the source for that date; that alone doesn't mean the market "
        "was closed. No-transaction is 0 because the source has no marker for it (a no-trade day simply has "
        "no row). Anomaly flags are statistical outliers kept as data, not removed."
    )
    st.dataframe(health.day_status, hide_index=True, width="stretch")

    st.subheader("Quarantined rows by reason")
    st.caption("Rows that failed a hard data contract; they are never used as prices or actuals.")
    if health.quarantine_by_reason.empty:
        st.info("No rows quarantined.")
    else:
        st.dataframe(health.quarantine_by_reason, hide_index=True)

    st.subheader("Widespread-gap dates")
    st.caption("Dates on which at least half the tracked series were unreported.")
    st.dataframe(health.widespread_gap_dates, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Mandi price forecasts", layout="wide")
    st.title("Mandi price forecasts: West Bengal tomato and onion")
    data = load_dashboard_data()
    st.caption(
        f"Data read at {data['read_at']:%Y-%m-%d %H:%M} IST (refreshed at most every "
        f"{CACHE_SECONDS // 60} minutes). Source: Agmarknet via data.gov.in."
    )

    scorecard_tab, forecasts_tab, health_tab, incidents_tab = st.tabs(
        ["Scorecard", "Forecasts", "Data health", "Incidents"]
    )
    with scorecard_tab:
        show_scorecard_tab(data)
    with forecasts_tab:
        show_forecasts_tab(data)
    with health_tab:
        show_health_tab(data)
    with incidents_tab:
        st.markdown((PROJECT_ROOT / "docs" / "incidents.md").read_text(encoding="utf-8"))


main()
