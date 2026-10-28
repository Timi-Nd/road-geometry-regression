# Experimental GIS-length imputation

## What this experiment does

This experiment estimates missing `Shape__Length` using the existing C++
multiple regression engine with `SEGMENT_LENGTH_KM` and `SPEEDLIMIT` as
predictors. These are exploratory predictors: speed-limit coding, units and
availability at imputation time are unconfirmed. Recorded length and GIS
length are closely related measurements, so this is not a road deterioration
model. GIS-length units remain unconfirmed. See the
[dataset investigation](dataset_investigation.md).

**The current snapshot has no missing `Shape__Length` values.** No genuine
observations were imputed. The reported accuracy comes solely from
deliberately hiding known values. Predictions for a different dataset must
not be represented as measurements or assumed to have these errors.

## Optional reproduction

After building the existing Python extension from the repository root:

```sh
python src/impute_length.py --output-dir results/imputation
python -m unittest discover -s tests -p "test_impute_length.py"
```

The [configuration](../config/imputation.json) fixes the target, features,
seed, fractions, QR rank tolerance and residual-quantile level. Input and
configuration SHA-256 hashes, settings, model coefficients, calibration
radius, counts, metrics and library versions are recorded in
[imputation_report.json](imputation/imputation_report.json).
The command overwrites only its three named output files in the chosen
directory, not the source CSV; `--csv`, `--config`, and `--output-dir` may
select other paths. Do not select an output directory containing files you
want to retain under those names.

The separate [masked-validation file](imputation/masked_validation.csv) marks
every artificially hidden row and includes its actual value **solely for
validation**, both regression and median predictions, and interval bounds.
The [imputed-values file](imputation/imputed_values.csv) contains one row per
genuinely blank target, with an explicit `is_imputed` flag, estimate, bounds,
or a reason for not imputing. It currently has only its header. `source_row`
is the zero-based data-row position excluding the header, tied to the input
hash; `ROADNAME` and `NID` are included as additional identifiers. The source
dataset is never modified.

Only blank or whitespace-only target fields are considered missing. Nonnumeric
or non-finite *nonblank* targets are reported as invalid, not silently
imputed. Inputs must have finite predictors, positive speed limit and
nonblank identifiers. Rows that cannot be imputed remain in the output with
`is_imputed=False` and no estimate. Zero speed is excluded because its meaning
is unknown, not because it has been confirmed missing.

## Masking and evaluation

Of 25,415 input rows, 25,258 have a known target and eligible predictors;
157 are excluded by the positive-speed policy. With seed 2027, known rows
are assigned by transitive `ROADNAME`/`NID` connected groups: 15,156 train,
5,068 calibration and 5,034 masked test. No `ROADNAME` or `NID` crosses
these partitions. They may still be geographically related; a control-section
name is not necessarily a whole highway.

The regression model and median baseline are fitted using training targets
only. Test targets are hidden from fitting and calibration, then restored
solely to score predictions. The baseline uses the training median, 895.02
source units. Earlier phases explored the full snapshot and compared related
models, so this is not a completely untouched research-history holdout. The
mask simulates known values missing by group, not the unknown mechanism
responsible for real missingness.

| Masked test method | MAE | RMSE | R² |
|---|---:|---:|---:|
| Regression | 101.56 | 244.06 | 0.996250 |
| Training median | 1,426.08 | 4,067.46 | -0.041431 |

This regression is substantially better than the median on this masked
sample, but its closely related length predictor makes that result unsurprising.
Generalization to genuinely missing data, particularly when predictor
availability or missingness mechanisms differ, remains unestablished.

## Experimental uncertainty

For the masked test experiment, an empirical symmetric interval uses the
90th percentile of the **calibration set's absolute residuals**, giving a
radius of 166.30 source units. On the masked test rows its observed coverage
is 87.56%, with mean width 332.60. This is a fixed-width descriptive
interval, **not a guaranteed 90% prediction interval**: group dependence,
geographic distribution shifts, heterogeneous residuals and nonrandom real
missingness can invalidate exchangeability, and it does not adjust for feature
uncertainty. Its bounds can be negative, and no clipping is applied.

For actual blank targets, the separate final model fits all eligible known
rows except the calibration group (20,190 rows), then recalculates the
calibration residual quantile (radius 184.78). Its calibration rows are not
in its training fit. Since this final model also uses the formerly masked
test labels for training, the reported masked-test coverage and errors
evaluate the **validation model**, not an independent evaluation of the
refitted final model. There were no actual blanks to estimate in this
snapshot; the final-model interval has not been validated on actual missing
targets.
