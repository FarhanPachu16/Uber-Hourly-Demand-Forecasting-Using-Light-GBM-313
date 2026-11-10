"""Helpers for the Uber demand forecast app.

Pure pandas / numpy (no Streamlit imports), so the numbers shown in the app can be tested on their own.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
HORIZONS = {"1 hour ahead": 1, "24 hours ahead": 24}


def load_forecasts(horizon_hours):
    """Load the saved August test-set forecasts for one horizon (1 or 24 hours)."""
    df = pd.read_parquet(DATA_DIR / f"august_forecasts_h{horizon_hours}.parquet")
    df = df.rename(columns={"trips": "actual", "last_week": "last_week_baseline"})
    df["hour"] = pd.to_datetime(df["hour"])
    # the zone id keeps labels unique (a few TLC zones share a name)
    df["zone_label"] = (df["Zone"].astype(str) + " (" + df["Borough"].astype(str)
                        + ") #" + df["zone_id"].astype(str))
    return df


def wape(actual, predicted):
    """Weighted absolute percentage error: total absolute error / total actual trips, in %."""
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    total = a.sum()
    return float("nan") if total == 0 else float(100 * np.abs(a - p).sum() / total)


def coverage(actual, lo, hi):
    """Share of actual values that fall inside [lo, hi], in %."""
    a = np.asarray(actual, float)
    inside = (a >= np.asarray(lo, float)) & (a <= np.asarray(hi, float))
    return float(100 * inside.mean()) if len(a) else float("nan")


def summary(df):
    """Headline numbers for any slice of the forecast table."""
    return {
        "trips": float(df["actual"].sum()),
        "model_wape": wape(df["actual"], df["forecast"]),
        "baseline_wape": wape(df["actual"], df["last_week_baseline"]),
        "coverage": coverage(df["actual"], df["lo80"], df["hi80"]),
    }


def zone_options(df):
    """Zones ordered from busiest to quietest (by total actual trips)."""
    return (df.groupby(["zone_id", "zone_label"], as_index=False)["actual"].sum()
              .sort_values("actual", ascending=False).reset_index(drop=True))


def zone_window(df, zone_id, start, days):
    """One zone, `days` days starting at midnight on `start`."""
    start = pd.Timestamp(start).normalize()
    end = start + pd.Timedelta(days=int(days))
    mask = (df["zone_id"] == zone_id) & (df["hour"] >= start) & (df["hour"] < end)
    return df.loc[mask].sort_values("hour")


def hour_snapshot(df, when):
    """All zones at one hour."""
    return df[df["hour"] == pd.Timestamp(when)]


def topn_capture(snapshot, n=15):
    """If drivers are sent to the n zones with the highest forecast, what % of the demand that the
    n truly busiest zones had do those zones actually hold?"""
    best_possible = snapshot.nlargest(n, "actual")["actual"].sum()
    chosen = snapshot.nlargest(n, "forecast")["actual"].sum()
    return float("nan") if best_possible == 0 else float(100 * chosen / best_possible)


def results_overview():
    """One row per horizon: model vs same-hour-last-week baseline on the whole test month."""
    rows = []
    for label, h in HORIZONS.items():
        d = load_forecasts(h)
        s = summary(d)
        rows.append({
            "Horizon": label,
            "Model WAPE %": round(s["model_wape"], 2),
            "Same hour last week WAPE %": round(s["baseline_wape"], 2),
            "Error reduction %": round(100 * (s["baseline_wape"] - s["model_wape"]) / s["baseline_wape"], 1),
            "80% interval coverage %": round(s["coverage"], 1),
        })
    return pd.DataFrame(rows)
