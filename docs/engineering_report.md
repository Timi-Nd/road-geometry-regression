# Engineering report: road geometry regression

## Outcome

The project now includes a multiple linear regression engine alongside the
original single-feature implementation. The experiment predicts GIS geometry
length from related road-inventory fields; it **does not predict road
deterioration**. The target's units and coordinate reference system remain
unconfirmed.

## Deliverables

| Deliverable | Location |
|---|---|
| C++ engine and API | [multiple_regression.cpp](../cpp/multiple_regression.cpp), [header](../cpp/multiple_regression.h), [original implementation](../cpp/regression.cpp) |
| Python bindings and build | [bindings](../cpp/multiple_regression_bindings.cpp), [setup.py](../setup.py), [requirements](../requirements.txt) |
| Automated tests and Ubuntu CI configuration | [tests](../tests/), [workflow](../.github/workflows/tests.yml) |
| Data investigation | [dataset investigation](../results/dataset_investigation.md), [audit script](../src/investigate_dataset.py) |
| Model experiments | [comparison report](../results/model_comparison.md), [script](../src/compare_models.py), [configuration](../config/model_comparison.json), [results](../results/model_comparison/model_comparison.json) |
| Notebook showcase | [single- and multiple-regression notebook](../notebooks/linear_regression.ipynb) |
| Interactive visualization | [dashboard](../results/model_comparison/dashboard.html), [generator](../src/visualize_models.py) |
| Experimental imputation | [imputation report](../results/imputation.md), [script](../src/impute_length.py), [configuration](../config/imputation.json), [flagged estimates](../results/imputation/imputed_values.csv) |
| Build, use and technical explanation | [README](../README.md), [technical guide](technical_guide.md) |

Build, test and reproduction commands, output overwrite behavior and dependency
details are documented in the [technical guide](technical_guide.md). Saved
experiment files include dataset and configuration hashes.

## Design choices

- The new engine uses column-scaled, unpivoted Householder QR followed by
  back-substitution rather than explicitly inverting `XᵀX`. It checks input
  dimensions, finite values, rank tolerance and numerical overflow. Unpivoted
  rank detection is order-dependent and is not an SVD condition guarantee.
- pybind11 converts NumPy-compatible inputs to the C++ representation. The
  mathematical fit and prediction remain in C++; the original Python
  functions and notebook workflow are preserved.
- Comparison uses common train, validation and test partitions for the
  pre-specified feature sets. Connected components of `ROADNAME` and `NID`
  avoid sharing those identifiers between grouped partitions; the random-row
  split is retained as a leakage-prone diagnostic.
- Imputation estimates, provenance and artificial-mask results are stored
  separately from the input. A calibration residual quantile supplies an
  exploratory fixed-width interval, not guaranteed coverage.

## Findings

On the connected-group test split, held-out RMSE was **278.70** for length
alone, **262.03** with speed limit, and **262.00** with speed limit and lane
count, in unconfirmed target units. The small additional improvement from lane
count is not evidence of reliable generalization or causality. Earlier
full-dataset exploration means the test is not a completely untouched
research-history holdout. The fields describe related length measurements,
which helps explain the high R² without demonstrating broader road
prediction. See the [full comparison](../results/model_comparison.md).

The current dataset has **no missing `Shape__Length` targets**, so no real
values were imputed. On 5,034 artificially masked known targets, regression
MAE/RMSE were 101.56/244.06 versus 1,426.08/4,067.46 for a training-median
baseline. The calibration-based interval covered 87.56% of the masked test
sample; it is not a guaranteed 90% interval. See the
[imputation report](../results/imputation.md).

## Verification and limitations

- A fresh Windows MSVC extension build, standalone C++ suite with warnings
  treated as errors, and **32 Python tests** passed. The dashboard was checked
  in a browser. Experiment outputs were separately reproduced and checked.
- Ubuntu has **not** been verified: the local WSL startup probe stalled, and
  the configured CI workflow has not yet been observed running. Do not treat
  the workflow definition as a passed Linux build.
- A follow-up public metadata check found that the current
  [HIGHWAY_OFFICIAL layer](https://services3.arcgis.com/zcv98lgAl8xQ04cW/arcgis/rest/services/HIGHWAY_OFFICIAL/FeatureServer/0?f=pjson)
  declares Web Mercator (3857) and a `Shape__Length` field in projected
  metres. That supports the existing hypothesis but does not establish
  the historical CSV's precise source/export pipeline or its geometry CRS;
  [details](../results/dataset_investigation.md#current-public-service-metadata).
- The code uses unpinned Python dependencies. Exact floating-point results and
  runtime measurements can vary by environment; measurements are not
  controlled benchmarks.
- Speed-limit coding and predictor availability remain unconfirmed; 157
  zero-speed rows were excluded consistently. Connected groups do not
  guarantee geographic or highway-level independence. Error tails include
  influential long segments.
- Actual missing-target behavior and the final imputation model's uncertainty
  have not been independently evaluated. Artificial masking may not match the
  real missingness mechanism.

These limitations define the scope of the showcase, rather than a to-do list
for visitors. The saved results and documentation can be explored without
running or extending the project.
