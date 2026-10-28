# Road Geometry Regression

An engineering and data-science showcase using Saskatchewan highway data: **how well do recorded segment length and other available road attributes predict GIS geometry length (`Shape__Length`)?**

The original single-feature regression remains available. A separate C++ multiple-regression engine uses Householder QR and is exposed to Python through pybind11. Python scripts investigate the data, compare models and validate experimental imputation. Charts are interactive [Apache ECharts](https://echarts.apache.org/).

**Start here:** [model comparison](results/model_comparison.md) · [dataset investigation](results/dataset_investigation.md) · [imputation experiment](results/imputation.md) · [technical guide](docs/technical_guide.md)

> **Scope:** This project studies closely related road-length measurements, not road deterioration. The dataset contains no missing GIS-length targets, so imputation was evaluated by masking known values.

## What I found

On one connected-group test split, the following models were trained on the same eligible rows and evaluated against `Shape__Length`:

| Predictors | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Recorded segment length | 113.54 | 278.70 | 0.995619 |
| Length + speed limit | 102.03 | 262.03 | 0.996127 |
| Length + speed limit + lane count | 101.92 | 262.00 | 0.996128 |

Error values are in unconfirmed source units. Adding speed limit improved the score on this split; adding lane count made little further difference. The predictors and target include closely related length measurements, so these results do not demonstrate deterioration prediction or generalization to unseen highways. The split prevents shared road names and NIDs, but does not guarantee geographic independence. See the [full comparison](results/model_comparison.md) for validation results, residuals, timings and caveats.

The imputation experiment hid 5,034 known values: regression MAE/RMSE were 101.56/244.06, compared with 1,426.08/4,067.46 for a training-median baseline. **No actual records were imputed** because the target has no missing values in this snapshot. Details and uncertainty limitations are in the [imputation report](results/imputation.md).

### Original single-feature experiment

Fitted on 80% of the 25,415 segments and scored on the 20% held out (seed 42):

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Line with intercept | 146.8 | 233.5 | 0.9953 |
| Line through the origin | 135.0 | 263.0 | 0.9940 |

<<<<<<< HEAD
The two measurements are almost perfectly linearly related. Similar train and test scores on a random row split do not establish generalization to unseen highways. The slope of about 1,777 per km may reflect Web Mercator distortion, but the target's units and CRS have not been confirmed. Full details are in [results/findings.md](results/findings.md).
=======
The two measurements are almost perfectly linearly related, and train and test scores match, so the model is not overfitted. The data indicate `Shape__Length` is measured in Web Mercator units, which are stretched by `1 / cos(latitude)` (about 1.6 at 52°N). That finding is inferred from the geometry, not confirmed from the source's metadata. Full details are in [results/findings.md](results/findings.md).
>>>>>>> 14b376aa6decca357e0c005dcc3fb87f22f09a0c

This is a regression exercise on two length measurements. It does not predict road deterioration, safety, traffic or maintenance.

The multiple-model comparison above uses a different split and excludes zero-speed rows, so its scores are not directly comparable with this original experiment.

For the implementation, mathematical walkthrough and reproducibility details, see the [technical guide](docs/technical_guide.md).
The [engineering report](docs/engineering_report.md) summarizes deliverables, verification and remaining limitations.

## Explore the showcase

The saved reports and results can be viewed without running anything. The [notebook](notebooks/linear_regression.ipynb) presents the original single-feature analysis, a multiple-regression API example, the grouped comparison and the imputation findings. GitHub may not render the notebook's interactive JavaScript charts; the written results remain available.

For visitors who choose to run the notebook without a local installation:

- **GitHub Codespaces:** click **Code > Codespaces > Create codespace on main**. The configuration is intended to install dependencies and build the C++ module; open `notebooks/linear_regression.ipynb`. Availability and build success have not been independently verified.
- **Binder:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Timi-Nd/road-geometry-regression/main?urlpath=lab/tree/notebooks/linear_regression.ipynb) opens the notebook (the first launch can take several minutes). Availability and build success depend on the published `main` branch and those services.

The charts load Apache ECharts from a CDN, so the browser needs internet access.

## Optional: run it locally

```powershell
pip install -r requirements.txt
python setup.py build_ext --inplace   # on Windows, run from a Visual Studio x64 developer prompt
```

Then open `notebooks/linear_regression.ipynb` and run all cells. The build needs a C++17 compiler.

## Multiple regression C++ engine

The C++ engine in `cpp/multiple_regression.h` and `cpp/multiple_regression.cpp`
fits multiple predictors using Householder QR least squares. It is also exposed
through `regression_cpp`; the notebook demonstrates both the original
single-feature API and the multiple-regression API.

After building the extension, fit a model with a two-dimensional array of
observations and a one-dimensional target:

```python
import numpy as np
import regression_cpp

X = np.array([[0, 0], [1, 0], [0, 1], [1, 2]], dtype=float)
y = 4 + 2 * X[:, 0] - 3 * X[:, 1]
model = regression_cpp.fit_multiple_regression(
    X, y, fit_intercept=True, rank_tolerance=1e-12
)
print(model.intercept, model.coefficients)
print(model.fitted_predictions, model.residuals)
print(model.predict([[2, 3]]))
```

Inputs may be NumPy-compatible arrays or lists; non-contiguous arrays are
accepted. The result's array properties and predictions are independent NumPy
arrays. Invalid shapes, non-finite inputs, and rank-deficient designs raise
`ValueError`; arithmetic overflow raises `OverflowError`. Predictions require
the same number of feature columns, including for an empty prediction array.

The Python binding tests can be run after building the extension:

```sh
python -m unittest discover -s tests -p "test_multiple_regression.py"
```

Pass a row-major `Matrix` (one observation per row) and a target vector to
`road_geometry::fit_multiple_regression`. The result contains the intercept,
coefficients in feature order, fitted predictions, residuals (actual minus
predicted), and a `predict` method for new rows. `RegressionOptions` controls
intercept fitting and the rank tolerance. The default relative tolerance is
`1e-12`, and must be finite and strictly between zero and one.

The solver scales each design column to unit Euclidean norm before applying
Householder reflections, solves the triangular system, and transforms the
coefficients back to their original units. The rank tolerance is compared with
the norm remaining after preceding columns have been eliminated; therefore it
detects near-dependence relative to each column's scale, not a condition number.
It rejects non-finite input, ragged or incompatible dimensions, too few
observations, all-zero features, rank deficiency and non-finite calculation
results. A constant nonzero feature is rank deficient when fitting an intercept,
but is allowed when the intercept is disabled.

The standalone C++ tests can be run on Ubuntu with a C++17 compiler:

```sh
g++ -std=c++17 -Wall -Wextra -pedantic cpp/multiple_regression.cpp tests/multiple_regression_test.cpp -o /tmp/multiple_regression_test
/tmp/multiple_regression_test
```

On Windows, run from a Visual Studio x64 developer prompt:

```powershell
cl /EHsc /std:c++17 /W4 cpp\multiple_regression.cpp tests\multiple_regression_test.cpp /Fe:multiple_regression_test.exe
.\multiple_regression_test.exe
```

## Optional: build and test on Ubuntu

The project requires Python, the packages in `requirements.txt`, and a C++17
compiler such as GCC. On Ubuntu, install Python development headers and
`build-essential` if they are not already available. From the repository root:

```sh
python3 -m pip install -r requirements.txt
python3 setup.py build_ext --inplace
g++ -std=c++17 -Wall -Wextra -Werror -pedantic cpp/multiple_regression.cpp tests/multiple_regression_test.cpp -o /tmp/multiple_regression_test
/tmp/multiple_regression_test
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The C++ tests cover fitting, predictions, residuals, numerical scaling and
degenerate inputs. The Python tests additionally check QR solutions against
NumPy, residual orthogonality, rank thresholds, bindings, preprocessing,
metrics, visualization data, imputation masking, and file integrity. Matrix
operations are currently internal to the solver, so their behavior is tested
through least-squares solutions rather than a separate public matrix API.
Some visualization tests use the checked-in dataset and saved experiment files.
The [Ubuntu CI workflow](.github/workflows/tests.yml) is configured to run
these build and test commands on pushes, pull requests, and manual dispatch;
its execution has not been verified. The Windows build and tests passed in
the documented run.

## Interactive model comparison

The [grouped-test dashboard file](results/model_comparison/dashboard.html) is
an HTML artifact; GitHub's file viewer does not run it as an interactive
page. To explore it, clone or download the repository and open
`results/model_comparison/dashboard.html` from your computer in a browser.
Use its model selector to explore predicted versus actual values,
residuals, their distribution, MAE/RMSE and R², coefficients and geographic
error patterns. ECharts is loaded from a CDN, so the browser needs internet
access. Zoom and chart toolboxes allow inspection of dense regions and outliers.

The geographic plot shows the **first vertex** of each available test geometry
on coordinate axes, not a road map or segment centroid; the CRS is unconfirmed.
Colour indicates residual sign and uses a signed logarithmic magnitude.
Coefficient charts use separate, zero-inclusive axes because feature units
differ: their magnitudes are not comparable feature importance.

To regenerate the grouped dashboard from the saved experiment, or create one
for the leakage-prone random-row diagnostic split:

```sh
python src/visualize_models.py --output results/model_comparison/dashboard.html
python src/visualize_models.py --split random_row_diagnostic --output results/model_comparison/random_dashboard.html
python -m unittest discover -s tests -p "test_visualize_models.py"
```

Regeneration requires the saved model-comparison JSON and test-prediction
CSV; the comparison can be rerun if they are unavailable. The dashboard generation checks the dataset hash,
identifiers, fitted coefficients, test predictions and saved metrics, and
refuses inconsistent inputs. See the [comparison report](results/model_comparison.md)
for split limitations and caveats.

## Experimental imputation

The [imputation report](results/imputation.md) describes a reproducible
artificial-masking experiment and median baseline. The current dataset has
no missing GIS-length targets; no actual values were imputed. To rerun:

```sh
python src/impute_length.py --output-dir results/imputation
python -m unittest discover -s tests -p "test_impute_length.py"
```

Results and any estimates are written separately from the input CSV. The
experimental intervals are empirical, not guaranteed prediction intervals.

## What the notebook does

1. Loads all 25,415 rows of `data/road_segments.csv` and checks both columns for missing, non-numeric or infinite values (none found).
2. Plots segment length against `Shape__Length` (hover for the road name and km range; switch between linear and log axes).
3. Fits the line `Shape__Length = b0 + b1 * SEGMENT_LENGTH_KM` by ordinary least squares.
4. Splits 80/20, fits on the training rows and scores the test rows.
5. Plots the test residuals, compares predicted with actual test values, and lists the largest misses.
6. Compares the fit with a through-origin model.
7. Demonstrates the multiple-regression API and reproduces the connected-group model comparison.
8. Summarizes the artificial-mask imputation results and their limitations.

## Project layout

| Path | Purpose |
|---|---|
| `notebooks/linear_regression.ipynb` | Original analysis, multiple-regression example and experiment results |
| `cpp/regression.cpp` | C++ least-squares fit, predictions, MAE, RMSE, R² (pybind11) |
| `cpp/multiple_regression.h`, `cpp/multiple_regression.cpp` | Standalone multiple regression engine |
| `cpp/multiple_regression_bindings.cpp` | NumPy-compatible pybind11 interface for multiple regression |
| `tests/multiple_regression_test.cpp` | Standalone C++ multiple regression tests |
| `tests/test_multiple_regression.py` | Python binding and backward-compatibility tests |
| `setup.py` | Builds the C++ extension as the `regression_cpp` module |
| `src/charts.py` | Apache ECharts chart helpers |
| `src/visualize_models.py` | Reproducible interactive dashboard from saved test predictions |
| `src/compare_models.py`, `src/impute_length.py` | Model comparison and experimental imputation pipelines |
| `config/` | Version-controlled experiment settings |
| `docs/technical_guide.md` | Algorithm, integration, reproduction and limitations |
| `results/dataset_investigation.md`, `results/model_comparison.md`, `results/imputation.md` | Data, model and imputation findings |
| `data/road_segments.csv` | Road-segment dataset (25,415 rows × 33 columns) |
| `docs/road_segments_methodology.md` | Where the dataset comes from and how it was prepared |
| `results/findings.md` | Original single-feature results, units hypothesis, and largest residuals |
| `requirements.txt` | Python dependencies |
| `.devcontainer/`, `binder/` | Configs for Codespaces and Binder |

## Notes and limitations

- Segments are adjacent parts of the same highways, so a random split does not show how the model would do on completely unseen highways.
- The units of `Shape__Length` are unconfirmed; the charts label it "source units".
- The [dataset investigation](results/dataset_investigation.md) profiles candidate predictors, missing values and leakage risks; it does not run model experiments.
- The [model comparison](results/model_comparison.md) evaluates the pre-specified one-, two- and three-feature models with random-row and connected-group splits. Reproduction commands are optional and documented in the report.

## Data provenance and reuse

The included [dataset](data/road_segments.csv) is a project-prepared provincial-highway snapshot, assembled from multiple sources for learning and exploration. It is not the original Ministry-supplied file, which is not included in this repository. The [methodology](docs/road_segments_methodology.md) documents the sources and preparation, including surface attributes described as derived from a separately supplied Ministry layer.

The current public highway service declares Web Mercator and a projected-metre shape-length field, but the historical export's exact units and geometry CRS remain unverified. See the [metadata investigation](results/dataset_investigation.md#current-public-service-metadata).

No project code licence is currently provided in this repository. The source terms and permissions applicable to sharing the prepared dataset, including any derived surface attributes, have not been confirmed. Its educational purpose alone does not establish those permissions.
