"""Reproduce the descriptive dataset and feature-availability audit."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
NUMERIC_COLUMNS = (
    "KM_FROM",
    "KM_TO",
    "SEGMENT_LENGTH_KM",
    "Shape__Length",
    "SPEEDLIMIT",
    "NBRLANES",
    "SURFACE_MATCH_PCT",
    "GOV_DOMINANT_PCT",
)
CORRELATION_COLUMNS = (
    "SEGMENT_LENGTH_KM",
    "Shape__Length",
    "KM_FROM",
    "KM_TO",
    "SPEEDLIMIT",
    "NBRLANES",
)


def numeric_profile(series):
    """Distinguish blank, nonnumeric and non-finite fields before summarizing."""
    blank = series.str.strip().eq("")
    values = pd.to_numeric(series.mask(blank), errors="coerce")
    finite = pd.Series(np.isfinite(values), index=series.index)
    usable = values[finite]
    percentiles = (0, 0.01, 0.25, 0.5, 0.75, 0.99, 1)
    return {
        "blank": int(blank.sum()),
        "nonnumeric": int((~blank & values.isna()).sum()),
        "nonfinite": int((values.notna() & ~finite).sum()),
        "finite": int(finite.sum()),
        "quantiles": {
            str(q): float(usable.quantile(q)) if not usable.empty else None
            for q in percentiles
        },
    }


def geometry_profile(series):
    """Check JSON path structure and coordinate bounds, not CRS or GIS accuracy."""
    counts = {"blank": 0, "invalid": 0, "valid": 0}
    bounds = None
    for raw in series:
        if not raw.strip():
            counts["blank"] += 1
            continue
        try:
            geometry = json.loads(raw)
            paths = geometry["paths"]
            if not isinstance(paths, list) or not paths:
                raise ValueError("paths must be a non-empty list")
            points = []
            for path in paths:
                if not isinstance(path, list) or not path:
                    raise ValueError("path must contain vertices")
                for vertex in path:
                    if (
                        not isinstance(vertex, list)
                        or len(vertex) < 2
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(value)
                            for value in vertex[:2]
                        )
                    ):
                        raise ValueError("vertex must contain two finite coordinates")
                    points.append(vertex[:2])
            if not points:
                raise ValueError("geometry has no vertices")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            counts["invalid"] += 1
            continue
        for x, y in points:
            if bounds is None:
                bounds = [x, y, x, y]
            else:
                bounds = [min(bounds[0], x), min(bounds[1], y),
                          max(bounds[2], x), max(bounds[3], y)]
        counts["valid"] += 1
    return {**counts, "coordinate_bounds": bounds}


def profile_dataset(frame):
    """Describe the full snapshot; these statistics are not a held-out evaluation."""
    required = set(NUMERIC_COLUMNS) | {"geometry_json", "ROADNAME", "NID"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {', '.join(missing_columns)}")
    if frame.empty:
        raise ValueError("CSV contains no observations")

    fields = {}
    for column in frame.columns:
        series = frame[column]
        fields[column] = {
            "blank": int(series.str.strip().eq("").sum()),
            "literal_None": int(series.str.strip().eq("None").sum()),
            "distinct_nonblank": int(series[~series.str.strip().eq("")].nunique()),
        }
    numeric = frame[list(NUMERIC_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    corr = numeric[list(CORRELATION_COLUMNS)].corr()
    correlations = {
        column: {
            other: None if pd.isna(value) else float(value)
            for other, value in corr[column].items()
        }
        for column in corr
    }
    length = numeric["SEGMENT_LENGTH_KM"]
    # This ratio is descriptive; it does not prove the target's units or projection.
    ratio = (numeric["Shape__Length"] / (1000 * length)).replace(
        [np.inf, -np.inf], np.nan
    )
    ratio = ratio.where(length > 0)
    difference = (numeric["KM_TO"] - numeric["KM_FROM"] - length).abs()
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "fields": fields,
        "numeric": {name: numeric_profile(frame[name]) for name in NUMERIC_COLUMNS},
        "correlations": correlations,
        "speed_limit_counts": frame["SPEEDLIMIT"].value_counts().to_dict(),
        "lane_counts": frame["NBRLANES"].value_counts().to_dict(),
        "zero_speed_count": int(numeric["SPEEDLIMIT"].eq(0).sum()),
        "km_difference_max_abs": float(difference.max()) if difference.notna().any() else None,
        "length_ratio_median": float(ratio.median()) if ratio.notna().any() else None,
        "length_ratio_outside_1_5_to_2": int(((ratio < 1.5) | (ratio > 2)).sum()),
        "geometry": geometry_profile(frame["geometry_json"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "road_segments.csv")
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.csv.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    frame = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    report = {"sha256": digest.hexdigest(), **profile_dataset(frame)}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
