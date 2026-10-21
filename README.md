# Road Geometry Linear Regression

A simple linear regression on Saskatchewan highway data: **how well does a road segment's recorded length (km) predict its GIS geometry length (`Shape__Length`)?**

The regression mathematics (least-squares fit, predictions, MAE, RMSE, R²) is written from scratch in C++ and called from a Jupyter notebook through pybind11. Charts are interactive [Apache ECharts](https://echarts.apache.org/).

## Key results

Fitted on 80% of the 25,415 segments and scored on the 20% held out (seed 42):

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Line with intercept | 146.8 | 233.5 | 0.9953 |
| Line through the origin | 135.0 | 263.0 | 0.9940 |

The two measurements are almost perfectly linearly related, and train and test scores match, so the model is not overfitted. The data indicate `Shape__Length` is measured in Web Mercator units, which are stretched by `1 / cos(latitude)` (about 1.6 at 52°N). That finding is inferred from the geometry, not confirmed from the source's metadata. Full details are in [results/findings.md](results/findings.md).

This is a regression exercise on two length measurements. It does not predict road deterioration, safety, traffic or maintenance.

## Run it online

No installation needed:

- **GitHub Codespaces:** click **Code > Codespaces > Create codespace on main**. The environment installs the dependencies and builds the C++ module automatically; then open `notebooks/linear_regression.ipynb` and run all cells.
- **Binder:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Timi-Nd/road-geometry-regression/main?urlpath=lab/tree/notebooks/linear_regression.ipynb) (the first launch can take several minutes while the image builds).

The charts load Apache ECharts from a CDN, so the browser needs internet access.

## Run it locally

```powershell
pip install -r requirements.txt
python setup.py build_ext --inplace   # on Windows, run from a Visual Studio x64 developer prompt
```

Then open `notebooks/linear_regression.ipynb` and run all cells. The build needs a C++17 compiler.

## What the notebook does

1. Loads all 25,415 rows of `data/road_segments.csv` and checks both columns for missing, non-numeric or infinite values (none found).
2. Plots segment length against `Shape__Length` (hover for the road name and km range; switch between linear and log axes).
3. Fits the line `Shape__Length = b0 + b1 * SEGMENT_LENGTH_KM` by ordinary least squares.
4. Splits 80/20, fits on the training rows and scores the test rows.
5. Plots the test residuals, compares predicted with actual test values, and lists the largest misses.
6. Compares the fit with a through-origin model.

## Project layout

| Path | Purpose |
|---|---|
| `notebooks/linear_regression.ipynb` | The analysis, start to finish |
| `cpp/regression.cpp` | C++ least-squares fit, predictions, MAE, RMSE, R² (pybind11) |
| `setup.py` | Builds the C++ extension as the `regression_cpp` module |
| `src/charts.py` | Apache ECharts chart helpers |
| `data/road_segments.csv` | Road-segment dataset (25,415 rows × 33 columns) |
| `docs/road_segments_methodology.md` | Where the dataset comes from and how it was prepared |
| `results/findings.md` | Written results, the units finding, and the largest residuals |
| `requirements.txt` | Python dependencies |
| `.devcontainer/`, `binder/` | Configs for Codespaces and Binder |

## Notes and limitations

- Segments are adjacent parts of the same highways, so a random split does not show how the model would do on completely unseen highways.
- The units of `Shape__Length` are unconfirmed; the charts label it "source units".
