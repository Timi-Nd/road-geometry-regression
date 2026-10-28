# Findings: predicting `Shape__Length` from `SEGMENT_LENGTH_KM`

Data: `data/road_segments.csv`, all 25,415 rows. Every row had finite values for both columns, so none were excluded. Model: simple linear regression with the maths in C++ (`cpp/regression.cpp`). Notebook: `notebooks/linear_regression.ipynb`.

## Results

Fit on all rows (descriptive only): `Shape__Length = -149.68 + 1777.03 * SEGMENT_LENGTH_KM`, R² = 0.9955.

Held-out evaluation (80/20 random split, seed 42; 20,332 train, 5,083 test):

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| OLS with intercept (-152.7 + 1779.8x) | 146.8 | 233.5 | 0.9953 |
| Through origin (1751.7x) | 135.0 | 263.0 | 0.9940 |

- Train and test scores are almost identical on this random split (train R² 0.9955); adjacent segments and shared road identifiers limit what this says about generalization.
- Over seeds 1, 2, 3, 7 and 42 the intercept model had the lower test RMSE every time (by about 20 to 30). The through-origin model had the lower MAE in the seed-42 split; MAE was not compared across the other seeds.
- Errors are in `Shape__Length` units (see below), not necessarily metres.

## Units hypothesis

`Shape__Length / (SEGMENT_LENGTH_KM * 1000)` has a median of 1.61, not 1. Testing this against `geometry_json` for 25,411 rows with geometry (4 are missing): for segments longer than 0.5 km, `Shape__Length` divided by `1000 * SEGMENT_LENGTH_KM / cos(latitude)` averages 0.999 (std 0.010), and the correlation between the measured ratio and `1 / cos(latitude)` is 0.96.

This pattern is consistent with **Web Mercator (EPSG:3857) projected metres**, which are stretched by approximately `1 / cos(latitude)` (about 1.6 at 52°N). The current public highway service identifies its shape-length field as projected metres, but the historical CSV's precise export and CRS are unconfirmed. See the [metadata investigation](dataset_investigation.md#current-public-service-metadata). Projected metres are not necessarily ground metres.

Consequences:
- The slope of about 1,777 per km is broadly consistent with projection distortion; the snapshot's provenance does not establish this as the cause.
- Latitude varies between segments and is not a model input, so it is the likely main source of the residual error (not measured). This is out of scope for simple linear regression.

## Largest residuals

The biggest misses on the full-data fit are all long segments (over 30 km):

| Road | Segment km | Residual (full-data fit) |
|---|---|---|
| 9050899LUA | 85.6 | +15,811 |
| 9050600MUA | 65.6 | +10,130 |
| 9050800MUA | 51.5 | +9,120 |
| 9550400MUA | 100.8 | +8,335 |

Their ratios are about 1.9 to 1.96, larger than the typical 1.61, which is consistent with those segments lying at higher latitudes (not checked). 228 of the 25,415 segments have a ratio outside 1.5 to 2.0, including some below 0.2 and above 4; their causes have not been established.