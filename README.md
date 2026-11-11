# Hourly Uber Demand Forecasting for NYC Taxi Zones

Forecasts how many Uber trips will start in each of **261 NYC taxi zones** in each hour, **1 hour and 24 hours ahead**,
using about **121 million** trip records from the NYC Taxi & Limousine Commission (January to August 2026).

On a held-out month (August 2026) the model reduces forecast error by **28.5% (1 hour ahead)** and **16.1% (24 hours ahead)**
compared with the strongest simple baseline, *same hour last week*. It also produces calibrated 80% prediction intervals
and was used in a simulated driver-allocation study.

**Live app:** [uber-hourly-demand-forecasting-using-light-gbm.streamlit.app](https://uber-hourly-demand-forecasting-using-light-gbm.streamlit.app/)  |  **Notebook:** `notebooks/uber_demand_forecasting.ipynb`  |  **Data preparation:** `prepare_data.py`

## Results at a glance (August 2026, held-out test month)

| Horizon | Same hour last week (best baseline) | LightGBM model | Error reduction | 80% interval coverage |
|---|---|---|---|---|
| 1 hour ahead | 16.44% WAPE | **11.76% WAPE** | **28.5%** | 81.2% |
| 24 hours ahead | 16.44% WAPE | **13.79% WAPE** | **16.1%** | 83.3% |

WAPE = total absolute error / total actual trips. It is used instead of MAPE because many zone-hours have zero or very few trips.

Simulated driver allocation (see below): unmet demand is **25.0% lower** (1 hour) and **17.6% lower** (24 hours) than allocating by the
same-hour-last-week baseline.

## Problem and data

- **Question:** how many Uber trips will start in each zone in each hour, early enough for a ride-hailing operator to position drivers?
- **Source:** [NYC TLC High Volume For-Hire Vehicle trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), monthly parquet files, January to August 2026. Uber trips are the records with `hvfhs_license_num = HV0003`.
- **Size:** about 121.4 million Uber trips (about 499,000 a day), aggregated to **1,522,152 zone-hours** (261 zones x 5,832 hours). Each row is the number of trips that started in one zone in one hour.
- **Processing:** the raw monthly files (roughly 480-540 MB each) do not fit in memory on a small machine, so they were aggregated with DuckDB (out-of-core SQL) to hourly counts per pickup zone. Zone-hours with no trips were added back as zeros (57,390 rows, 3.8%). Zone names come from the TLC taxi zone lookup.
- **Data quality:** 1,824 zone-hours (0.12%) had a negative average wait time (a request timestamp after the pickup timestamp). Those wait values were set to missing; the trip counts were kept.

## Method

- **Target:** trips started in a zone in an hour. One **global** LightGBM model for all zones, with the zone as a feature, using Poisson loss because the target is a non-negative count.
- **Features (all known at forecast time):** hour of day, day of week, weekend flag, US federal holiday flag, airport flag, zone and borough; lagged trips (1, 2, 3, 24, 48 and 168 hours back where the horizon allows); rolling 24-hour and 168-hour averages; the previous wait time. Every history feature is at least as old as the forecast horizon, so nothing from the future leaks in.
- **Baselines:** last value, same hour yesterday, same hour last week.
- **Split (time-based, no shuffling):** train January to June, validate July (model settings, interval calibration), test August (evaluated once, after all choices were fixed).
- **Model settings:** learning rate 0.05, 63 leaves, at least 100 samples per leaf, 80% feature and row subsampling. Training ran to the 2,000-round cap; validation loss was still improving very slightly at that point.
- **Prediction intervals:** 10% and 90% quantile LightGBM models, widened by a conformal adjustment computed separately for five groups of expected volume, calibrated on all of July.

## Detailed results (August 2026 test)

### Accuracy

| 1 hour ahead | MAE | RMSE | WAPE % |
|---|---|---|---|
| Last value | 15.45 | 26.66 | 20.28 |
| Same hour yesterday | 19.17 | 38.68 | 25.15 |
| Same hour last week | 12.53 | 24.00 | 16.44 |
| **LightGBM (Poisson)** | **8.96** | **16.29** | **11.76** |

| 24 hours ahead | MAE | RMSE | WAPE % |
|---|---|---|---|
| Last value (= same hour yesterday) | 19.17 | 38.68 | 25.15 |
| Same hour last week | 12.53 | 24.00 | 16.44 |
| **LightGBM (Poisson)** | **10.51** | **20.01** | **13.79** |

The baseline rows are identical in both tables because they do not depend on the horizon.

### By segment (WAPE %, model vs same hour last week)

| Segment | 1 h model | 1 h last week | 24 h model | 24 h last week |
|---|---|---|---|---|
| All zones | 11.8 | 16.4 | 13.8 | 16.4 |
| Airports | 14.6 | 20.3 | 17.3 | 20.3 |
| Top 20 zones | 10.6 | 15.4 | 13.6 | 15.4 |
| All other zones | 12.1 | 16.8 | 13.9 | 16.8 |
| Weekends | 11.3 | 15.7 | 13.8 | 15.7 |

### Prediction intervals (80% nominal, calibrated on July)

| Coverage % | 1 h raw | 1 h calibrated | 24 h raw | 24 h calibrated |
|---|---|---|---|---|
| All zones | 78.6 | 81.2 | 81.2 | 83.3 |
| Airports | 60.9 | 88.1 | 75.9 | 87.0 |
| Top 20 zones | 78.9 | 81.3 | 83.1 | 85.5 |
| All other zones | 78.6 | 81.2 | 81.0 | 83.1 |
| Weekends | 79.8 | 82.3 | 82.7 | 84.7 |

Average interval width after calibration: 29.8 trips (1 h) and 36.7 trips (24 h), against an average of 76.2 trips per zone-hour. The calibrated intervals are slightly conservative, especially at airports.

### Simulated driver allocation

In each hour, assume the city has as many available drivers as there were trips that hour, and split them across zones in proportion to a forecast. A trip is "served" if a driver is in its zone. A perfect forecast would serve 100%.
**Assumptions:** supply equals total demand, drivers can be repositioned instantly and for free, one trip per driver-hour. This is a simulation, not a measured business outcome; it compares how useful different forecasts are for positioning.

| Allocation based on | Trips served | Unmet |
|---|---|---|
| Even split across zones | 66.4% | 33.6% |
| Same hour yesterday | 90.9% | 9.1% |
| Same hour last week | 92.8% | 7.2% |
| LightGBM, 24 h ahead | 94.0% | 6.0% (17.6% lower than last week) |
| LightGBM, 1 h ahead | 94.6% | 5.4% (25.0% lower than last week) |
| Perfect forecast (limit) | 100% | 0% |

## What validation taught me

- **The July 4 weekend was the hardest stretch.** Daily WAPE on 24-hour forecasts was 18.9% (Fri 3 July), 29.9% (Sat 4 July), 22.1% (Sun 5 July) and 19.5% (Mon 6 July), against roughly 11-16% on most other days of July (the 18-19 July weekend was also higher, at 17.5% and 19.0%). The training months contain no July 4, so the model could not learn it. It also hurt the same-hour-last-week baseline, which is why the model's relative gain was larger in July (about 38% at 1 hour, about 24% at 24 hours) than in August, the cleaner test month.
- **Weather did not help.** Hourly ERA5 weather (temperature, precipitation, snowfall, wind) from Open-Meteo was added as features. With identical settings, 24-hour WAPE on July went from 14.95% to 15.23%, and from 16.38% to 17.89% in rainy hours (both versions stopped at the 500-round cap, with the same settings). Likely reasons (not verified): training covers only January to June, so July heat is outside the training range; one coarse grid point stands in for the whole city; and observed weather is not a forecast. Weather is not used in the final model.

## Limitations

- Only **completed** Uber trips are in the data, so this forecasts fulfilled demand, not total demand (cancelled and unmatched requests are missing).
- **Eight months of one year.** The model has never seen July 4, Labor Day, Thanksgiving or Christmas, and has seen only one winter-to-summer cycle.
- No **flight, event or live traffic** data. Airports are the hardest zones, and flight arrivals drive airport demand.
- The holiday flag uses the US federal calendar's observed dates: Independence Day 2026 was flagged on Friday 3 July, not on Saturday 4 July itself.
- The August test month has no major holiday.
- The conformal intervals are calibrated on one month and are slightly conservative.
- The app shows **saved test-set forecasts**; the trained models are not served live.

## Repository

```
.
├── README.md
├── app.py                 # Streamlit app
├── forecast_utils.py      # data loading and metrics used by the app
├── requirements.txt       # packages the app needs
├── prepare_data.py        # downloads the TLC files and builds the hourly-by-zone table (DuckDB)
├── requirements-pipeline.txt   # packages for prepare_data.py and the notebook
├── data/
│   ├── august_forecasts_h1.parquet    # 1-hour-ahead forecasts, intervals, baseline for August
│   └── august_forecasts_h24.parquet   # 24-hour-ahead
└── notebooks/
    └── uber_demand_forecasting.ipynb  # features, models, validation, August test, simulation
                                       # (starts from the table that prepare_data.py writes)
```

Run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reproducing from raw data

1. `pip install -r requirements-pipeline.txt`
2. `python prepare_data.py` downloads the monthly `fhvhv_tripdata_2026-MM.parquet` files and `taxi_zone_lookup.csv` from the TLC (one file at a time, about 0.5 GB each, deleted after use), aggregates each month to hourly Uber trips per pickup zone with DuckDB, adds zero-trip zone-hours, attaches zone names, and writes `uber_hourly_full.parquet`.
3. Open `notebooks/uber_demand_forecasting.ipynb`. If needed, set `DATA_PATH` in the first code cell to where `uber_hourly_full.parquet` is. Run it top to bottom with `H = 1`, then again with `H = 24`. It covers features, baselines, the model, validation on July, the one-time August test, the allocation simulation, saving the forecasts used by the app, and (as an appendix) the weather experiment.

## Credits and notes

- Trip data: NYC Taxi and Limousine Commission. TLC does not guarantee the accuracy or completeness of the data. This project is not affiliated with Uber or the TLC.
- Weather experiment only: weather data by [Open-Meteo.com](https://open-meteo.com/), based on ERA5 reanalysis (Copernicus Climate Change Service).
