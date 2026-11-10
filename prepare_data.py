"""Build the hourly Uber-trips-by-zone table used by the notebook and the app.

Steps
1. Download each monthly NYC TLC High Volume For-Hire Vehicle parquet file, one at a time.
2. With DuckDB, keep Uber trips only (hvfhs_license_num = 'HV0003') and count trips by pickup hour and
   pickup zone. The raw files (about 0.5 GB each) are never loaded into pandas, so a small machine is enough.
3. Combine the months, add the zone-hours that had no trips as zero rows, and attach zone names.

Output: uber_hourly_full.parquet (one row per hour and zone) in the work folder.

Usage:
    python prepare_data.py                      # January to August 2026, files in the current folder
    python prepare_data.py --workdir data_work  # keep the downloads and outputs in another folder
"""
import argparse
import os

import duckdb
import numpy as np
import pandas as pd
import requests

BASE = "https://d37ci6vzurychx.cloudfront.net"
DEFAULT_MONTHS = [f"2026-{m:02d}" for m in range(1, 9)]
UBER_LICENSE = "HV0003"
AVG_COLS = ["avg_wait_sec", "avg_trip_miles", "avg_fare"]


def download(url, path, chunk=1024 * 1024):
    """Stream a file to disk and check its size against what the server reports."""
    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    expected = int(head.headers.get("Content-Length", 0))
    if expected and os.path.exists(path) and os.path.getsize(path) == expected:
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for part in r.iter_content(chunk):
                f.write(part)
    got = os.path.getsize(path)
    if expected and got != expected:
        raise RuntimeError(f"Incomplete download: {got} of {expected} bytes")


def aggregate_month(con, raw, out):
    """Shrink one raw monthly file to Uber trips per pickup hour and zone."""
    con.execute(f"""
        COPY (
            SELECT date_trunc('hour', pickup_datetime)                         AS hour,
                   PULocationID                                                AS zone_id,
                   count(*)                                                    AS trips,
                   avg(date_diff('second', request_datetime, pickup_datetime)) AS avg_wait_sec,
                   avg(trip_miles)                                             AS avg_trip_miles,
                   avg(base_passenger_fare)                                    AS avg_fare
            FROM '{raw}'
            WHERE hvfhs_license_num = '{UBER_LICENSE}'
            GROUP BY 1, 2
        ) TO '{out}' (FORMAT parquet)
    """)


def build_panel(frames, zones, start, end):
    """Combine monthly hourly tables into one complete hour x zone table.

    frames: list of DataFrames with hour, zone_id, trips and the AVG_COLS columns.
    zones:  the TLC taxi zone lookup (LocationID, Borough, Zone, service_zone).
    start, end: first and last hour to keep (inclusive).
    """
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["hour"] >= start) & (df["hour"] <= end)]

    # An hour and zone can appear in two monthly files if trips near a month edge were split across them.
    # Merge those rows, weighting the averages by the number of trips.
    if df.duplicated(["hour", "zone_id"]).any():
        for c in AVG_COLS:
            df[c] = df[c] * df["trips"]
        df = df.groupby(["hour", "zone_id"], as_index=False).sum()
        for c in AVG_COLS:
            df[c] = df[c] / df["trips"]

    # A zone with no trips in an hour has no row yet: add it back with zero trips
    all_hours = pd.date_range(start, end, freq="h")
    all_zones = np.sort(df["zone_id"].unique())
    grid = pd.MultiIndex.from_product([all_hours, all_zones], names=["hour", "zone_id"])
    full = df.set_index(["hour", "zone_id"]).reindex(grid)
    full["trips"] = full["trips"].fillna(0).astype(int)
    full = full.reset_index()

    # pandas reads the text "N/A" in service_zone as missing; the feature step turns it into "Unknown"
    full = (full.merge(zones[["LocationID", "Borough", "Zone", "service_zone"]],
                       left_on="zone_id", right_on="LocationID", how="left")
                .drop(columns="LocationID"))
    return full


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", nargs="+", default=DEFAULT_MONTHS, help="months as YYYY-MM")
    ap.add_argument("--workdir", default=".", help="where downloads and outputs are written")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    tmp = os.path.join(args.workdir, "duck_tmp")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")

    frames = []
    for m in args.months:
        out = os.path.join(args.workdir, f"uber_hourly_{m}.parquet")
        if not os.path.exists(out):
            raw = os.path.join(args.workdir, f"fhvhv_tripdata_{m}.parquet")
            try:
                download(f"{BASE}/trip-data/fhvhv_tripdata_{m}.parquet", raw)
                aggregate_month(con, raw, out)
            finally:
                if os.path.exists(raw):
                    os.remove(raw)          # the raw file is about 0.5 GB; keep only the small table
        frames.append(pd.read_parquet(out))
        print("done", m)

    zones_path = os.path.join(args.workdir, "taxi_zone_lookup.csv")
    download(f"{BASE}/misc/taxi_zone_lookup.csv", zones_path)
    zones = pd.read_csv(zones_path)

    start = pd.Timestamp(f"{args.months[0]}-01")
    end = pd.Timestamp(f"{args.months[-1]}-01") + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
    panel = build_panel(frames, zones, start, end)

    out_path = os.path.join(args.workdir, "uber_hourly_full.parquet")
    panel.to_parquet(out_path)
    print(f"hours: {panel['hour'].nunique()} | zones: {panel['zone_id'].nunique()} | rows: {len(panel):,}")
    print(f"Uber trips in total: {int(panel['trips'].sum()):,}")
    print("saved", out_path)


if __name__ == "__main__":
    main()
