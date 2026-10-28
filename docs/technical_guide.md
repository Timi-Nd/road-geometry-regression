# Technical guide: road geometry regression

## What changed

The [original C++ implementation](../cpp/regression.cpp) fits a line with
one predictor using centered closed-form ordinary least squares. It also
supports a through-origin fit, prediction and MAE/RMSE/R². Those functions
remain in `regression_cpp` and the
[notebook](../notebooks/linear_regression.ipynb) retains their original analysis.

The [multiple-regression engine](../cpp/multiple_regression.cpp) is a
separate implementation of

```text
y_hat = intercept + beta[0] * x[0] + ... + beta[p-1] * x[p-1]
```

with an optional intercept. It fits any positive number of numerical features
subject to having at least as many observations as fitted parameters and a
full-rank design. It returns fitted predictions, residuals (`actual -
predicted`), and a model that predicts new rows. Both implementations are
compiled into the same Python extension; the old API is unchanged. Later
notebook sections demonstrate the new API and the connected-group comparison.

## How the QR solver maps to the code

For readers new to regression: a **feature** is an input column, the
**target** is the value being estimated, and a **coefficient** tells how the
prediction changes for a one-unit feature increase while other inputs stay
fixed. The **intercept** is the prediction when all features are zero.
A **residual** is the actual target minus its prediction. These are fitted
associations, not proof that changing a feature causes the target to change.

In `fit_multiple_regression`, each row of `features` is one observation.
When an intercept is requested, the code prepends a column of ones to form a
design matrix `A`. It scales every column by its Euclidean norm using
`std::hypot`, producing `A_scaled`. An all-zero column is rejected.

For each column `k`, a normalized Householder vector is formed from the
remaining column entries. Its sign is chosen to avoid cancellation. The
reflection `H = I - 2vvᵀ` is applied in place to the remaining design columns
and to the target vector. After all reflections, the leading square block is
upper triangular `R`, and the corresponding transformed target contains the
first entries of `Qᵀy`. The code solves `R gamma = Qᵀy` by back-substitution,
then divides `gamma` by the saved column norms to recover coefficients in the
original units. Predictions are formed using the original features.

QR avoids forming and inverting `AᵀA`; forming normal equations squares the
condition number and can amplify numerical error. Reflections preserve
Euclidean norms in exact arithmetic. This implementation uses unpivoted
Householder QR, not a rank-revealing pivoted QR or SVD. At each step it
compares the remaining norm of a unit-scaled design column with
`rank_tolerance` (default `1e-12`); the result can depend on column order and
is not a condition-number guarantee.

For a concrete example, the [README](../README.md) fits rows
`[[0,0], [1,0], [0,1], [1,2]]` to targets `[4,6,1,0]`. The fitted
intercept is approximately 4 and coefficients are approximately `[2,-3]`.
For a new row `[2,3]`, the prediction is `4 + 2*2 - 3*3 = -1`.
Floating-point results can differ slightly from these exact values. The
[binding tests](../tests/test_multiple_regression.py) check this example and
compare other solutions with NumPy's least-squares reference.

Input validation rejects empty training data, ragged or incompatible shapes,
non-finite values, insufficient observations, all-zero features and
rank-deficient designs. A nonzero constant feature is dependent on an
intercept but can be valid without one. A finite tolerance strictly between
zero and one is required. Overflow is reported explicitly. These checks do
not make an ill-conditioned but accepted problem statistically reliable.

## Python and C++ boundary

[`setup.py`](../setup.py) builds `regression_cpp` with C++17 and pybind11.
[`multiple_regression_bindings.cpp`](../cpp/multiple_regression_bindings.cpp)
accepts NumPy-compatible inputs, converts them to contiguous double arrays,
checks dimensions, and copies them into the C++ matrix/vector representation.
The numerical fit and prediction run in C++; the binding releases the Python
GIL during those operations. Result properties and predictions are copied to
independent NumPy arrays. C++ invalid-argument and overflow exceptions surface
as `ValueError` and `OverflowError`, respectively.

```python
import regression_cpp

model = regression_cpp.fit_multiple_regression(
    [[0, 0], [1, 0], [0, 1], [1, 2]],
    [4, 6, 1, 0],
    fit_intercept=True,
    rank_tolerance=1e-12,
)
print(model.intercept, model.coefficients)
print(model.fitted_predictions, model.residuals)
print(model.predict([[2, 3]]))
```

`predict` expects a two-dimensional array with the fitted number of columns,
including for an empty batch. Inputs must be available at prediction time.

## Optional: build, test and reproduce

From the repository root, install [requirements](../requirements.txt) and
build with a C++17 compiler:

```sh
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
python -m unittest discover -s tests -p "test_*.py" -v
```

On Windows, use a Visual Studio x64 developer prompt. Ubuntu also needs a
compiler and Python development headers; the [README](../README.md#optional-build-and-test-on-ubuntu)
gives the standalone C++ test command. The Windows MSVC build and 32 Python
tests passed in the documented run. Ubuntu execution was not
verified locally because WSL startup stalled; the
[Ubuntu workflow](../.github/workflows/tests.yml) is configured but its
execution has not been verified here.

Commands below write outputs to the named directories; existing files with
those output names are overwritten:

```sh
python src/investigate_dataset.py
python src/compare_models.py --output-dir results/model_comparison
python src/visualize_models.py --output results/model_comparison/dashboard.html
python src/impute_length.py --output-dir results/imputation
```

The model and imputation settings, including seeds, are in
[model_comparison.json](../config/model_comparison.json) and
[imputation.json](../config/imputation.json). Result JSON records input and
configuration hashes and library versions. Reproduction requires the same
input snapshot and compatible dependencies; packages are not pinned, and
timings vary between runs. The dashboard loads ECharts from a CDN and needs
browser internet access. Generated predictions and imputed estimates are
stored separately from the input CSV.

## Data, models and findings

The target is `Shape__Length`; its units and CRS are unconfirmed.
`SEGMENT_LENGTH_KM` is derived from `KM_TO - KM_FROM`, and is closely
related to the target. Using all three with an intercept produces a dependent
design. Geometry-derived length would directly reconstruct the target and
would be leakage in this prediction task.

The current public highway service declares Web Mercator and a shape-length
field in projected metres, but the historical CSV's precise export and ETL
are not verified. See the [metadata investigation](../results/dataset_investigation.md#current-public-service-metadata)
for the source and the distinction between the service and this snapshot.

The [dataset investigation](../results/dataset_investigation.md) records
missingness, distributions, correlations and field caveats. Speed limit and
lane count were selected as provisional numeric attributes, not because of
held-out scores. Their coding and deployment-time availability remain
unconfirmed; 157 zero-speed rows are excluded consistently from all
comparison models. The dataset lacks historical condition, maintenance,
traffic and climate outcomes needed for deterioration prediction.

The [model comparison](../results/model_comparison.md) evaluates A (length),
B (length and speed), and C (length, speed and lanes). On the connected-group
test split, RMSE was 278.70, 262.03 and 262.00 source units, respectively.
The improvement from C over B is very small; a single split does not establish
reliable improvement or causality. Groups prevent shared `ROADNAME` and `NID`
across partitions, but may still be geographically related. The random-row
split shares identifiers and is a diagnostic only. Earlier full-data
exploration also means this is not a completely untouched research-history
test set.

## Imputation

The [imputation experiment](../results/imputation.md) masks known
`Shape__Length` values by connected group, trains on other groups and compares
predictions to the hidden values and a training-median baseline. Masked-test
MAE/RMSE were 101.56/244.06 for regression and 1,426.08/4,067.46 for the
median. The source snapshot has **no missing target values**, so there were
no genuine imputations.

An interval based on the 90th percentile of calibration absolute residuals
covered 87.56% of the masked test values. It is an empirical fixed-width
diagnostic, not a guaranteed prediction interval. The final model for actual
missing values is refitted separately; its uncertainty is not independently
validated on real missing values.

Historical target CRS and field definitions, geographic generalization and
performance under genuine missingness remain unconfirmed. This snapshot does
not include the dated condition, maintenance, traffic and environmental data
needed for deterioration prediction.
