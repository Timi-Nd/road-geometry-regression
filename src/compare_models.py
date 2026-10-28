"""Compare pre-specified road-length regression models."""

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import regression_cpp as regression


TARGET = "Shape__Length"
NUMERIC = ["SEGMENT_LENGTH_KM", "SPEEDLIMIT", "NBRLANES", TARGET]
IDENTIFIERS = ["ROADNAME", "NID"]
DEFAULT_MODELS = {
    "A": ["SEGMENT_LENGTH_KM"],
    "B": ["SEGMENT_LENGTH_KM", "SPEEDLIMIT"],
    "C": ["SEGMENT_LENGTH_KM", "SPEEDLIMIT", "NBRLANES"],
}


def load_config(path):
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    if config["models"] != DEFAULT_MODELS:
        raise ValueError("model definitions must match the pre-specified A/B/C comparison")
    train = config["train_fraction"]
    validation = config["validation_fraction"]
    if not (0 < train < 1 and 0 < validation < 1 and train + validation < 1):
        raise ValueError("split fractions must be positive and sum to less than one")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0 < config["rank_tolerance"] < 1:
        raise ValueError("rank tolerance must be between zero and one")
    if not isinstance(config["timing_repetitions"], int) or config["timing_repetitions"] < 1:
        raise ValueError("timing repetitions must be a positive integer")
    return config


def prepare_data(frame):
    """Keep the same usable rows for every model and report exclusions."""
    missing = sorted(set(NUMERIC + IDENTIFIERS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("dataset is empty")
    numeric = frame[NUMERIC].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric).all(axis=1)
    missing_ids = frame[IDENTIFIERS].apply(
        lambda col: col.isna() | col.astype(str).str.strip().isin(["", "None"])
    ).any(axis=1)
    # Zero speed has an unknown meaning; exclude it consistently, not as confirmed missing data.
    nonpositive_speed = numeric["SPEEDLIMIT"] <= 0
    valid = finite & ~missing_ids & ~nonpositive_speed
    stats = {
        "input_rows": len(frame),
        "nonfinite_or_nonnumeric": int((~finite).sum()),
        "missing_identifiers": int(missing_ids.sum()),
        "nonpositive_speed": int(nonpositive_speed.sum()),
        "excluded_union": int((~valid).sum()),
        "eligible": int(valid.sum()),
    }
    clean = frame.loc[valid, IDENTIFIERS].copy()
    clean[NUMERIC] = numeric.loc[valid, NUMERIC]
    # Retain original zero-based row positions for traceable saved predictions.
    clean.insert(0, "source_row", np.flatnonzero(valid.to_numpy()))
    clean.reset_index(drop=True, inplace=True)
    if len(clean) < 3:
        raise ValueError("at least three eligible observations are required")
    return clean, stats


def connected_groups(frame):
    """Group rows linked by either identifier, including chains of shared IDs.

    For example, if A shares a road name with B and B shares an NID with C,
    all three must remain in the same partition.
    """
    parent = list(range(len(frame)))
    size = [1] * len(frame)

    # Union-find tracks connected rows efficiently; each root identifies a group.
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]

    for column in IDENTIFIERS:
        first = {}
        for i, value in enumerate(frame[column]):
            if value in first:
                union(i, first[value])
            else:
                first[value] = i
    return np.array([find(i) for i in range(len(frame))], dtype=np.int64)


def make_splits(frame, config, grouped):
    """Return row positions for train, validation and test using a fixed seed."""
    rng = np.random.default_rng(config["seed"])
    if grouped:
        groups = connected_groups(frame)
        unique = np.unique(groups)
        if len(unique) < 3:
            raise ValueError("at least three connected groups are required")
        # Assign whole groups in random order. Leave at least one group for each
        # remaining partition, even if a large group crosses a row-count threshold.
        shuffled = rng.permutation(unique)
        counts = {int(group): int(np.count_nonzero(groups == group)) for group in unique}
        group_part = {}
        cumulative = 0
        stage = "train"
        validation_groups = 0
        cuts = (config["train_fraction"] * len(frame),
                (config["train_fraction"] + config["validation_fraction"]) * len(frame))
        for i, group in enumerate(shuffled):
            remaining = len(shuffled) - i
            if stage == "train" and (cumulative >= cuts[0] or remaining == 2):
                stage = "validation"
            if stage == "validation" and validation_groups and (
                cumulative >= cuts[1] or remaining == 1
            ):
                stage = "test"
            group_part[int(group)] = stage
            if stage == "validation":
                validation_groups += 1
            cumulative += counts[int(group)]
        labels = np.array([group_part[int(group)] for group in groups])
    else:
        # This random-row split is a diagnostic: related roads may cross partitions.
        order = rng.permutation(len(frame))
        n_train = int(len(frame) * config["train_fraction"])
        n_validation = int(len(frame) * config["validation_fraction"])
        labels = np.empty(len(frame), dtype="<U10")
        labels[order[:n_train]] = "train"
        labels[order[n_train:n_train + n_validation]] = "validation"
        labels[order[n_train + n_validation:]] = "test"
    parts = {name: np.flatnonzero(labels == name) for name in ("train", "validation", "test")}
    if any(len(part) == 0 for part in parts.values()):
        raise ValueError("split produced an empty partition; inspect group sizes or fractions")
    return parts


def overlap_report(frame, parts):
    """Count identifiers shared between each pair of partitions."""
    report = {}
    for column in IDENTIFIERS:
        sets = {name: set(frame.iloc[index][column]) for name, index in parts.items()}
        report[column] = {
            "train_validation": len(sets["train"] & sets["validation"]),
            "train_test": len(sets["train"] & sets["test"]),
            "validation_test": len(sets["validation"] & sets["test"]),
        }
    return report


def evaluate(actual, predicted):
    """Calculate error metrics; residuals are actual minus predicted."""
    residual = actual - predicted
    r2 = float(regression.r2(actual, predicted))
    return {
        "n": int(len(actual)),
        "mae": float(regression.mae(actual, predicted)),
        "rmse": float(regression.rmse(actual, predicted)),
        "r2": r2 if np.isfinite(r2) else None,
        "residual_mean": float(np.mean(residual)),
        "residual_median": float(np.median(residual)),
        "residual_p05": float(np.quantile(residual, 0.05)),
        "residual_p95": float(np.quantile(residual, 0.95)),
        "residual_max_abs": float(np.max(np.abs(residual))),
    }


def fit_and_predict(train_x, train_y, features, tolerance):
    """Fit the original one-feature API or the C++ multiple-feature API."""
    start = time.perf_counter()
    if len(features) == 1:
        intercept, slope = regression.fit_ols(train_x[:, 0], train_y)
        coefficients = np.array([slope])

        def predict(x):
            return regression.predict(x[:, 0], intercept, slope)
    else:
        model = regression.fit_multiple_regression(
            train_x, train_y, rank_tolerance=tolerance
        )
        intercept, coefficients = model.intercept, model.coefficients
        predict = model.predict
    fit_seconds = time.perf_counter() - start
    return intercept, coefficients, predict, fit_seconds


def compare(frame, config):
    """Evaluate every predefined model on the same partitions within each split."""
    results = {}
    predictions = []
    for split_name, grouped in (("random_row_diagnostic", False), ("connected_group", True)):
        parts = make_splits(frame, config, grouped)
        overlaps = overlap_report(frame, parts)
        if grouped and any(value for pairs in overlaps.values() for value in pairs.values()):
            raise RuntimeError("grouped split shares road or NID across partitions")
        split_result = {"counts": {key: len(value) for key, value in parts.items()},
                        "identifier_overlaps": overlaps, "models": {}}
        for model_name, features in config["models"].items():
            x = frame[features].to_numpy(dtype=float)
            y = frame[TARGET].to_numpy(dtype=float)
            train = parts["train"]
            intercept, coefficients, predict, fit_seconds = fit_and_predict(
                x[train], y[train], features, config["rank_tolerance"]
            )
            # This condition number describes scaling/dependence, not the QR rank test.
            design = np.column_stack((np.ones(len(train)), x[train]))
            norms = np.linalg.norm(design, axis=0)
            condition = float(np.linalg.cond(design / norms))
            if not np.isfinite(condition):
                raise ValueError("training design has a non-finite condition number")
            model_result = {
                "features": features,
                "intercept": float(intercept),
                "coefficients": dict(zip(features, map(float, coefficients))),
                "unit_column_condition_number": condition,
                "training_seconds": fit_seconds,
                "partitions": {},
            }
            for name, indices in parts.items():
                durations = []
                for _ in range(config["timing_repetitions"]):
                    start = time.perf_counter()
                    predicted = predict(x[indices])
                    durations.append(time.perf_counter() - start)
                metrics = evaluate(y[indices], predicted)
                metrics["prediction_seconds_median"] = float(np.median(durations))
                model_result["partitions"][name] = metrics
                # Save held-out predictions separately; never overwrite source observations.
                if name == "test":
                    for idx, value in zip(indices, predicted):
                        row = frame.iloc[idx]
                        predictions.append({
                            "split": split_name, "partition": name, "model": model_name,
                            "source_row": int(row["source_row"]), "ROADNAME": row["ROADNAME"],
                            "NID": row["NID"], "actual": float(y[idx]), "predicted": float(value),
                            "residual": float(y[idx] - value),
                        })
            split_result["models"][model_name] = model_result
        results[split_name] = split_result
    return results, predictions


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "road_segments.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "model_comparison.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    frame = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    clean, quality = prepare_data(frame)
    result, predictions = compare(clean, config)
    metadata = {
        "dataset_sha256": sha256(args.csv), "config_sha256": sha256(args.config),
        "config": config, "quality": quality,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "timing_note": "Single fit and median prediction wall-clock seconds, including Python/C++ conversion; machine-dependent.",
        "splits": result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "model_comparison.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, allow_nan=False)
        output.write("\n")
    with (args.output_dir / "model_predictions.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Compared {quality['eligible']} eligible rows; results written to {args.output_dir}")


if __name__ == "__main__":
    main()
