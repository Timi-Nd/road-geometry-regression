"""Experimental, separately recorded imputation of missing GIS lengths."""

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

import compare_models as comparison


ROOT = Path(__file__).resolve().parent.parent
TARGET = "Shape__Length"
FEATURES = ["SEGMENT_LENGTH_KM", "SPEEDLIMIT"]
FIELDS = ["source_row", "ROADNAME", "NID", "target", "status", "reason",
          "is_imputed", "prediction", "interval_lower", "interval_upper"]
MASK_FIELDS = ["source_row", "ROADNAME", "NID", "target", "is_artificially_masked",
               "actual_for_validation_only", "prediction", "baseline_prediction",
               "interval_lower", "interval_upper"]


def load_config(path):
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if config["target"] != TARGET or config["features"] != FEATURES:
        raise ValueError("the experiment requires the predefined target and features")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    train, calibration = config["train_fraction"], config["validation_fraction"]
    if not 0 < train < 1 or not 0 < calibration < 1 or train + calibration >= 1:
        raise ValueError("invalid training and calibration fractions")
    if not 0 < config["rank_tolerance"] < 1:
        raise ValueError("invalid rank tolerance")
    if not 0 < config["interval_quantile"] < 1:
        raise ValueError("interval quantile must be between zero and one")
    return config


def prepare(frame):
    """Separate blank targets, invalid values and rows eligible for fitting."""
    required = {TARGET, *FEATURES, "ROADNAME", "NID"}
    if required - set(frame.columns):
        raise ValueError(f"missing required columns: {sorted(required - set(frame.columns))}")
    if frame.empty:
        raise ValueError("dataset is empty")
    frame = frame.copy().reset_index(drop=True)
    frame.insert(0, "source_row", np.arange(len(frame)))
    raw_target = frame[TARGET].fillna("").astype(str).str.strip()
    missing = raw_target.eq("")
    target = pd.to_numeric(raw_target.mask(missing), errors="coerce")
    # A nonblank invalid value is not a blank: do not silently replace it.
    invalid_target = ~missing & ~np.isfinite(target)
    features = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    finite_features = pd.Series(np.isfinite(features).all(axis=1), index=frame.index)
    positive_speed = features["SPEEDLIMIT"] > 0
    valid_ids = pd.Series(True, index=frame.index)
    for column in ("ROADNAME", "NID"):
        valid_ids &= ~frame[column].fillna("").astype(str).str.strip().isin(["", "None"])
    valid_features = finite_features & positive_speed & valid_ids
    known = ~missing & ~invalid_target & valid_features
    clean = frame.loc[known, ["source_row", "ROADNAME", "NID"]].copy()
    clean[FEATURES] = features.loc[known, FEATURES]
    clean[TARGET] = target.loc[known]
    clean.reset_index(drop=True, inplace=True)
    if len(clean) < 3:
        raise ValueError("at least three eligible known target observations are required")
    stats = {
        "input_rows": len(frame),
        "missing_target": int(missing.sum()),
        "invalid_nonblank_target": int(invalid_target.sum()),
        "eligible_known": int(known.sum()),
        "ineligible_known_or_invalid": int((~missing & ~known).sum()),
        "missing_with_valid_predictors": int((missing & valid_features).sum()),
        "missing_with_invalid_predictors_or_ids": int((missing & ~valid_features).sum()),
    }
    return frame, clean, features, missing, valid_features, stats


def fit(frame, config):
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame[TARGET].to_numpy(dtype=float)
    return comparison.regression.fit_multiple_regression(
        x, y, rank_tolerance=config["rank_tolerance"]
    )


def model_details(model):
    return {"intercept": float(model.intercept),
            "coefficients": dict(zip(FEATURES, map(float, model.coefficients)))}


def interval_radius(model, calibration, quantile):
    """Use a calibration residual quantile as a descriptive interval radius."""
    predicted = model.predict(calibration[FEATURES].to_numpy(dtype=float))
    residuals = np.abs(calibration[TARGET].to_numpy(dtype=float) - predicted)
    return float(np.quantile(residuals, quantile))


def run(frame, config):
    """Evaluate artificial masking, then estimate genuinely blank eligible targets."""
    source, known, feature_values, missing, valid_features, stats = prepare(frame)
    parts = comparison.make_splits(known, config, grouped=True)
    overlaps = comparison.overlap_report(known, parts)
    if any(v for counts in overlaps.values() for v in counts.values()):
        raise RuntimeError("identifier overlap across validation partitions")
    # Test targets are hidden from fitting and calibration, then used for scoring.
    train = known.iloc[parts["train"]]
    calibration = known.iloc[parts["validation"]]
    hidden = known.iloc[parts["test"]]
    model = fit(train, config)
    radius = interval_radius(model, calibration, config["interval_quantile"])
    predicted = model.predict(hidden[FEATURES].to_numpy(dtype=float))
    actual = hidden[TARGET].to_numpy(dtype=float)
    # A training-only median is a simple reference, not a second fitted regression.
    baseline = float(train[TARGET].median())
    baseline_predicted = np.full(len(hidden), baseline)
    validation = {
        "regression": comparison.evaluate(actual, predicted),
        "median": comparison.evaluate(actual, baseline_predicted),
        "median_value": baseline,
        "interval_radius": radius,
        "interval_empirical_coverage": float(np.mean(np.abs(actual - predicted) <= radius)),
        "interval_mean_width": 2 * radius,
        "model": model_details(model),
        "partition_counts": {name: len(indices) for name, indices in parts.items()},
        "identifier_overlaps": overlaps,
    }
    masked_records = []
    for i, row in enumerate(hidden.itertuples(index=False)):
        masked_records.append({
            "source_row": int(row.source_row), "ROADNAME": row.ROADNAME, "NID": row.NID,
            "target": TARGET, "is_artificially_masked": True,
            "actual_for_validation_only": float(actual[i]), "prediction": float(predicted[i]),
            "baseline_prediction": baseline, "interval_lower": float(predicted[i] - radius),
            "interval_upper": float(predicted[i] + radius),
        })

    # The final model may also use the formerly hidden test labels. Its interval
    # is recalibrated separately, so the masked-test score describes the earlier
    # validation model, not an independent evaluation of this final refit.
    final_train = known.iloc[np.concatenate([parts["train"], parts["test"]])]
    final_model = fit(final_train, config)
    final_radius = interval_radius(final_model, calibration, config["interval_quantile"])
    # Estimates and reasons are written separately; the original CSV stays unchanged.
    eligible_missing = missing & valid_features
    missing_predictions = final_model.predict(
        feature_values.loc[eligible_missing, FEATURES].to_numpy(dtype=float)
    )
    estimate_by_row = dict(zip(source.loc[eligible_missing, "source_row"], missing_predictions))
    imputed_records = []
    for row in source.loc[missing, ["source_row", "ROADNAME", "NID"]].itertuples(index=False):
        value = estimate_by_row.get(row.source_row)
        if value is None:
            imputed_records.append({
                "source_row": int(row.source_row), "ROADNAME": row.ROADNAME, "NID": row.NID,
                "target": TARGET, "status": "not_imputed", "reason": "invalid predictors or identifiers",
                "is_imputed": False, "prediction": "", "interval_lower": "", "interval_upper": "",
            })
        else:
            imputed_records.append({
                "source_row": int(row.source_row), "ROADNAME": row.ROADNAME, "NID": row.NID,
                "target": TARGET, "status": "imputed", "reason": "",
                "is_imputed": True, "prediction": float(value),
                "interval_lower": float(value - final_radius), "interval_upper": float(value + final_radius),
            })
    final = {"model": model_details(final_model), "training_rows": len(final_train),
             "calibration_rows": len(calibration), "interval_radius": final_radius,
             "imputed_rows": len(estimate_by_row)}
    return stats, validation, final, masked_records, imputed_records


def write_csv(path, fields, records):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "road_segments.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "imputation.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_paths = [args.output_dir / name for name in
                    ("imputation_report.json", "masked_validation.csv", "imputed_values.csv")]
    # Protect the input and configuration even when output paths resolve through aliases.
    protected = {args.csv.resolve(), args.config.resolve()}
    if any(path.resolve() in protected for path in output_paths):
        raise ValueError("output must not overwrite the input dataset or configuration")
    config = load_config(args.config)
    frame = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    stats, validation, final, masked, imputed = run(frame, config)
    report = {
        "dataset_sha256": comparison.sha256(args.csv), "config_sha256": comparison.sha256(args.config),
        "config": config, "quality": stats, "validation": validation, "final_imputation_model": final,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "interval_method": "Symmetric empirical calibration absolute-residual quantile; not a guaranteed prediction interval.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_paths[0].open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    write_csv(output_paths[1], MASK_FIELDS, masked)
    write_csv(output_paths[2], FIELDS, imputed)
    print(f"Validated {len(masked)} masked observations; imputed {final['imputed_rows']} "
          f"of {stats['missing_target']} missing targets. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
