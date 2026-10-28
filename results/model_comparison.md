# Road-length model comparison

## Scope and reproducibility

This is an exploratory comparison for predicting `Shape__Length` in
**unconfirmed source units**, not a deterioration forecast. The models were
specified before this experiment in
[`config/model_comparison.json`](../config/model_comparison.json), based on
the field semantics described in the
[dataset investigation](dataset_investigation.md). Earlier full-dataset
descriptive exploration included target correlations, so this is not a
strictly untouched test set in the broad research-history sense. We did not
change the feature sets or choose a model based on the test scores below.

To reproduce the results optionally, build the existing C++ extension from
the repository root and run:

```sh
python setup.py build_ext --inplace
python src/compare_models.py --output-dir results/model_comparison
python -m unittest discover -s tests -p "test_compare_models.py"
```

The script writes the complete configuration, dataset and config hashes,
software versions, metrics, condition estimates and timings to
[`model_comparison.json`](model_comparison/model_comparison.json); it writes
the actual and predicted values and residuals for both test partitions to
[`model_predictions.csv`](model_comparison/model_predictions.csv). Source
row numbers are zero-based data-row positions in the hashed CSV, excluding
its header; predictions are not observations and are not written into the
input dataset. Re-running overwrites those two files in the chosen output
directory. Timings are not deterministic.

The [grouped-test dashboard](model_comparison/dashboard.html) visualizes the
saved predictions and metrics. GitHub displays the HTML file rather than
running it; open a downloaded copy locally in a browser with internet access
for ECharts. To regenerate it optionally, run
`python src/visualize_models.py --output results/model_comparison/dashboard.html`
to regenerate it. The geographic plot uses the first coordinate in each
geometry (not its centroid) and a signed-log residual colour scale. Both
target units and coordinate CRS are unconfirmed. See the README for the
optional random-row dashboard and visualization tests.

The data snapshot has SHA-256
`97897e1c149900e28e0fe6b7e9d976f10200fc81096a2042a87b6aa63dca9313`.
This run used Python 3.14.6, NumPy 2.4.4 and pandas 3.0.1 on Windows.

## Eligibility, features and splits

We excluded the 157 rows with nonpositive `SPEEDLIMIT` for **all** models
because the meaning of the observed zeros is unconfirmed. This is a
conservative experimental policy, not a finding that they are missing.
No rows had non-finite required fields or missing `ROADNAME`/`NID`; 25,258
rows remained. Exclusion counts may overlap if applying the script to
another dataset. Speed-limit units and prediction-time availability, as well
as lane-count definitions and availability, remain to be verified; therefore
B and C are provisional, not deployment-ready. No imputation was applied.

| Model | Features | Fitting method |
|---|---|---|
| A | `SEGMENT_LENGTH_KM` | Existing C++ single-feature OLS with intercept |
| B | `SEGMENT_LENGTH_KM`, `SPEEDLIMIT` | New C++ Householder QR, with intercept |
| C | `SEGMENT_LENGTH_KM`, `SPEEDLIMIT`, `NBRLANES` | New C++ Householder QR, with intercept |

Two seed-42 splits use approximately 60% train, 20% validation and 20% test,
with identical partitions across A/B/C within each split. All fits use only
training rows. The validation and test sets are evaluated separately, and no
hyperparameters were tuned.

- **Connected-group split (primary):** groups are transitive connected
  components of rows sharing `ROADNAME` or `NID`, assigned as whole groups
  in random order until row-count thresholds are reached. Train, validation
  and test have 15,170, 5,042 and 5,046 rows. There are zero shared
  `ROADNAME` and zero shared `NID` values between every pair of partitions.
  This prevents those known identifiers from crossing splits, not all
  possible spatial or network relationships. `ROADNAME` can identify a
  control-section alignment rather than a whole highway; disjoint names do
  not prove unseen highways or geographic independence.
- **Random-row split (diagnostic only):** train, validation and test contain
  15,154, 5,051 and 5,053 rows. Train and test share 1,164 `ROADNAME` values
  and 469 `NID` values. It is intentionally leakage-prone and is not the
  primary independent-segment evaluation. The full overlap counts for all
  partition pairs are in the JSON.

The earlier notebook used an 80/20 split on all rows, while this comparison
uses a three-way split and excludes zero-speed rows. The scores are not
directly comparable.

## Results

MAE and RMSE are in target source units; R² is unitless. Residual means
actual minus predicted. A single split does not quantify uncertainty.

### Connected-group split

| Model | Partition | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| A | Train | 104.65 | 171.67 | 0.997006 |
| A | Validation | 101.35 | 143.93 | 0.997720 |
| A | Test | 113.54 | 278.70 | 0.995619 |
| B | Train | 92.38 | 161.23 | 0.997359 |
| B | Validation | 84.60 | 134.44 | 0.998011 |
| B | Test | 102.03 | 262.03 | 0.996127 |
| C | Train | 92.17 | 161.21 | 0.997360 |
| C | Validation | 84.36 | 134.40 | 0.998012 |
| C | Test | 101.92 | 262.00 | 0.996128 |

On this split, B's test RMSE is 5.98% below A's; C is only 0.013% below B.
Validation RMSE shows the same direction (143.93, 134.44 and 134.40).
The small additional difference for lane count does not establish a meaningful
general improvement; neither difference establishes causality or performance
on new highways. Feature meaning and prediction-time availability remain
unconfirmed, and the models are not deployment-ready.

### Random-row diagnostic

| Model | Partition | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| A | Train | 119.93 | 197.96 | 0.996421 |
| A | Validation | 117.19 | 182.32 | 0.997053 |
| A | Test | 118.89 | 181.77 | 0.997302 |
| B | Train | 103.96 | 185.18 | 0.996868 |
| B | Validation | 103.22 | 174.46 | 0.997301 |
| B | Test | 103.12 | 169.85 | 0.997644 |
| C | Train | 103.70 | 185.14 | 0.996869 |
| C | Validation | 103.04 | 174.45 | 0.997302 |
| C | Test | 102.86 | 169.82 | 0.997645 |

These are different test rows from the connected-group split, so the scores
cannot isolate the effect of grouping alone. The shared identifiers make the
random-row scores unsuitable as evidence of generalization to unseen groups.

## Residuals, coefficients and stability

On the connected-group test set, median residuals for A/B/C are respectively
46.10, 19.10 and 17.23; their 5th-to-95th percentile ranges are
[-270.30, 102.55], [-250.46, 152.66] and [-250.49, 152.69]. Residuals
remain asymmetric and have large tails. The largest absolute test residual
for A is 13,223.57 source units, on a segment with actual length 187,320.37;
removing that single observation *for diagnosis only* changes A's grouped
test RMSE from 278.70 to 207.43. It was not removed from any reported model
score. This indicates strong sensitivity to long segments, not proof that the
record is erroneous.

| Model | Intercept | Length coefficient | Speed coefficient | Lane coefficient |
|---|---:|---:|---:|---:|
| A | -101.32 | 1,728.05 | — | — |
| B | 96.52 | 1,736.88 | -2.58 | — |
| C | 74.07 | 1,736.92 | -2.57 | 10.71 |

Coefficients reflect feature units and correlations, not causal importance.
Nearly all records have two lanes, and the reported speed coefficient may
capture geographic or other confounding. The unit-column-normalized design
condition numbers for the grouped training data are 1.73, 7.74 and 19.53
for A, B and C respectively. They are descriptive SVD-based condition
estimates, not guarantees of numerical stability or the QR rank test itself.

For this run, grouped-split fit times for A/B/C were approximately 0.00008,
0.00309 and 0.00309 seconds; median prediction times for the roughly 5,046
test rows were 0.000018, 0.000209 and 0.000228 seconds. Fit was timed once;
prediction is the median of five calls. Times include Python/C++ conversion,
depend on machine and load, and are not a controlled benchmark. The A API and
the multiple-feature APIs also have different conversion paths.

The length fields are closely related measurements; the high R² does not
demonstrate deterioration prediction or establish the target's CRS. Results
on independent geographic or highway-level holdouts are not available.
