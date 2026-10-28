"""Render interactive ECharts dashboards from saved model comparison results."""

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import charts


ROOT = Path(__file__).resolve().parent.parent
MODELS = ("A", "B", "C")
SPLITS = ("connected_group", "random_row_diagnostic")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_results(results_dir, data_path, split):
    """Verify saved predictions against their source and metrics before charting."""
    with (results_dir / "model_comparison.json").open(encoding="utf-8") as source:
        metadata = json.load(source)
    if sha256(data_path) != metadata["dataset_sha256"]:
        raise ValueError("dataset SHA-256 does not match the saved experiment")
    predictions = pd.read_csv(
        results_dir / "model_predictions.csv",
        dtype={"ROADNAME": str, "NID": str},
        keep_default_na=False,
    )
    required = {"split", "partition", "model", "source_row", "ROADNAME", "NID",
                "actual", "predicted", "residual"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    data = pd.read_csv(data_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    result = metadata["splits"][split]
    chosen = predictions[(predictions["split"] == split) & (predictions["partition"] == "test")]
    model_rows = {}
    source_rows = None
    for model in MODELS:
        rows = chosen[chosen["model"] == model].copy()
        if len(rows) != result["counts"]["test"] or rows.empty:
            raise ValueError(f"{model} test prediction count does not match experiment metadata")
        for column in ("source_row", "actual", "predicted", "residual"):
            rows[column] = pd.to_numeric(rows[column], errors="raise")
        values = rows[["source_row", "actual", "predicted", "residual"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{model} predictions contain non-finite values")
        ids = rows["source_row"].to_numpy()
        if np.any(ids != np.floor(ids)) or np.any(ids < 0) or np.any(ids >= len(data)):
            raise ValueError(f"{model} contains invalid source row numbers")
        ids = ids.astype(int)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{model} contains duplicate source rows")
        if source_rows is not None and set(ids) != source_rows:
            raise ValueError("models do not share the same test observations")
        source_rows = set(ids)
        # A saved row number is meaningful only for the matching, hashed input CSV.
        original = data.iloc[ids]
        if not np.array_equal(rows["ROADNAME"].to_numpy(), original["ROADNAME"].to_numpy()):
            raise ValueError(f"{model} road identifiers do not match source rows")
        if not np.array_equal(rows["NID"].to_numpy(), original["NID"].to_numpy()):
            raise ValueError(f"{model} NIDs do not match source rows")
        actual = pd.to_numeric(original["Shape__Length"], errors="raise").to_numpy()
        if not np.allclose(rows["actual"], actual, rtol=1e-12, atol=1e-9):
            raise ValueError(f"{model} actual values do not match the dataset")
        model_metadata = result["models"][model]
        # Independently reconstruct predictions from the stored coefficients.
        reconstructed = np.full(len(rows), model_metadata["intercept"], dtype=float)
        for feature in model_metadata["features"]:
            feature_values = pd.to_numeric(original[feature], errors="raise").to_numpy(dtype=float)
            reconstructed += model_metadata["coefficients"][feature] * feature_values
        if not np.allclose(rows["predicted"], reconstructed, rtol=1e-10, atol=1e-8):
            raise ValueError(f"{model} predictions do not match the saved coefficients")
        residual = rows["actual"].to_numpy() - rows["predicted"].to_numpy()
        if not np.allclose(rows["residual"], residual, rtol=1e-12, atol=1e-9):
            raise ValueError(f"{model} residuals do not match predictions")
        metrics = result["models"][model]["partitions"]["test"]
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual * residual)))
        sst = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = None if sst == 0 else float(1 - np.sum(residual ** 2) / sst)
        for key, value in (("mae", mae), ("rmse", rmse), ("r2", r2)):
            expected = metrics[key]
            if (value is None) != (expected is None) or (
                value is not None and not np.isclose(value, expected, rtol=1e-9, atol=1e-9)
            ):
                raise ValueError(f"{model} {key} does not match saved test metrics")
        rows["geometry_json"] = original["geometry_json"].to_numpy()
        model_rows[model] = rows
    return result, model_rows


def first_coordinate(raw):
    """Return the first geometry vertex, not the road centroid or a verified CRS."""
    if not raw.strip():
        return None
    try:
        paths = json.loads(raw)["paths"]
        vertex = paths[0][0]
        x, y = vertex[0], vertex[1]
        if (isinstance(x, bool) or isinstance(y, bool)
                or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
                or not math.isfinite(x) or not math.isfinite(y)
                or not -180 <= x <= 180 or not -90 <= y <= 90):
            return None
        return float(x), float(y)
    except (ValueError, KeyError, TypeError, IndexError):
        return None


def scatter_option(x, y, x_label, y_label, title, subtitle, zero_line=False):
    option = charts._base(charts._title(title, subtitle), x_label, y_label)
    option["tooltip"] = {"trigger": "item", "valueFormatter": "function(value){return Number(value).toLocaleString();}"}
    series = {
        "type": "scatter", "name": "Test segments",
        "data": np.column_stack((x, y)).tolist(),
        "symbolSize": 5, "itemStyle": {"color": charts.BLUE, "opacity": 0.45},
    }
    if zero_line:
        series["markLine"] = {
            "silent": True, "symbol": "none", "label": {"show": False},
            "lineStyle": {"color": charts.RED}, "data": [{"yAxis": 0}],
        }
    option["series"] = [series]
    return option


def histogram_option(residuals, subtitle):
    counts, edges = np.histogram(residuals, bins=60)
    labels = [f"{edges[i]:.1f} to {edges[i + 1]:.1f}" for i in range(len(counts))]
    option = charts._base(charts._title("Residual distribution", subtitle),
                          "Residual bins (source units)", "Test segments")
    option["xAxis"] = {"type": "category", "data": labels, "name": "Residual bins (source units)",
                       "nameLocation": "middle", "nameGap": 40, "axisLabel": {"show": False}}
    option["tooltip"] = {"trigger": "axis", "axisPointer": {"type": "shadow"}}
    option["series"] = [{"type": "bar", "name": "Test segments", "data": counts.tolist(),
                         "itemStyle": {"color": charts.BLUE}}]
    return option


def accuracy_option(result):
    option = charts._base(
        charts._title("Held-out error comparison", "MAE and RMSE; lower is better. Source units unconfirmed."),
        "Model", "Error (source units)",
    )
    option["xAxis"] = {"type": "category", "data": list(MODELS), "name": "Model",
                       "nameLocation": "middle", "nameGap": 30}
    option["legend"] = {"top": 54, "right": 60}
    option["tooltip"] = {"trigger": "axis"}
    option["series"] = [
        {"name": metric.upper(), "type": "bar", "data": [
            result["models"][model]["partitions"]["test"][metric] for model in MODELS
        ], "itemStyle": {"color": color}}
        for metric, color in (("mae", charts.BLUE), ("rmse", charts.RED))
    ]
    return option


def r2_option(result):
    option = charts._base(
        charts._title("Held-out R² comparison",
                      "Higher is better; vertical axis is zoomed. Test rows differ by split."),
        "Model", "R²",
    )
    option["yAxis"]["scale"] = True
    option["xAxis"] = {"type": "category", "data": list(MODELS), "name": "Model",
                       "nameLocation": "middle", "nameGap": 30}
    option["tooltip"] = {"trigger": "axis"}
    option["series"] = [{
        "name": "R²", "type": "bar",
        "data": [result["models"][model]["partitions"]["test"]["r2"] for model in MODELS],
        "itemStyle": {"color": charts.BLUE},
    }]
    return option


def coefficient_option(model):
    """Give each coefficient its own zero-inclusive axis because units differ."""
    coefficients = {"Intercept": model["intercept"], **model["coefficients"]}
    option = {
        "animation": False,
        "title": charts._title("Fitted coefficients", "Separate axes: units differ. Magnitudes are not feature importance."),
        "tooltip": {"trigger": "item"},
        "grid": [], "xAxis": [], "yAxis": [], "series": [],
    }
    count = len(coefficients)
    for i, (name, value) in enumerate(coefficients.items()):
        option["grid"].append({"left": "9%", "right": "6%", "top": f"{19 + i * 75 / count}%",
                               "height": f"{55 / count}%"})
        option["xAxis"].append({"type": "value", "gridIndex": i,
                                "min": min(0, value * 1.1), "max": max(0, value * 1.1) if value else 1,
                                "axisLabel": {"fontSize": 10}})
        option["yAxis"].append({"type": "category", "gridIndex": i, "data": [name],
                                "axisLabel": {"show": False}})
        option["series"].append({
            "name": name, "type": "bar", "xAxisIndex": i, "yAxisIndex": i,
            "data": [value], "label": {"show": True, "position": "inside", "formatter": name},
            "itemStyle": {"color": charts.BLUE if value >= 0 else charts.RED},
        })
    return option


def geographic_option(rows, subtitle):
    points = []
    for raw, residual in zip(rows["geometry_json"], rows["residual"]):
        coordinate = first_coordinate(raw)
        if coordinate is not None:
            # Signed log colour retains the direction of errors while showing smaller errors.
            color_value = math.copysign(math.log1p(abs(residual)), residual)
            points.append([*coordinate, float(residual), color_value])
    missing = len(rows) - len(points)
    if not points:
        raise ValueError("no valid geographic coordinates for the selected test set")
    bound = max(abs(point[3]) for point in points) or 1
    option = charts._base(
        charts._title("Geographic pattern of residuals",
                      f"{subtitle} First vertex; {missing} missing or invalid geometries."),
        "Longitude-like coordinate (CRS unconfirmed)", "Latitude-like coordinate",
    )
    option["xAxis"]["scale"] = True
    option["yAxis"]["scale"] = True
    option["grid"]["top"] = 115
    option["visualMap"] = {"type": "continuous", "min": -bound, "max": bound,
                           "dimension": 3, "left": 95, "top": 66,
                           "orient": "horizontal", "text": ["Positive", "Negative"],
                           "inRange": {"color": ["#2563eb", "#f3f4f6", "#dc2626"]}}
    option["tooltip"] = {
        "trigger": "item",
        "formatter": (
            "function(p){var d=p.data;return 'X: '+d[0].toFixed(4)+'<br/>Y: '+d[1].toFixed(4)"
            "+'<br/>Residual (actual - predicted): '+d[2].toLocaleString();}"
        ),
    }
    option["series"] = [{"type": "scatter", "data": points, "symbolSize": 6,
                          "itemStyle": {"opacity": 0.7}}]
    return option, missing


def make_options(result, rows_by_model, split):
    output = {}
    for model, rows in rows_by_model.items():
        subtitle = f"Model {model} | {split} test | {len(rows):,} segments | source units unconfirmed"
        actual = rows["actual"].to_numpy(dtype=float)
        predicted = rows["predicted"].to_numpy(dtype=float)
        residual = rows["residual"].to_numpy(dtype=float)
        actual_option = charts.actual_predicted_option(
            actual, predicted, x_label="Actual (source units)", y_label="Predicted (source units)",
            title="Predicted versus actual", subtitle=subtitle,
        )
        actual_option["series"][0]["large"] = False
        geo, missing = geographic_option(rows, subtitle)
        output[model] = {
            "actual": actual_option,
            "residual": scatter_option(predicted, residual, "Predicted (source units)",
                                       "Residual (source units)", "Residuals versus predicted",
                                       subtitle, zero_line=True),
            "histogram": histogram_option(residual, subtitle),
            "accuracy": accuracy_option(result),
            "r2": r2_option(result),
            "coefficients": coefficient_option(result["models"][model]),
            "geography": geo,
            "missing_geometry": missing,
        }
    return output


def dashboard_html(options, split):
    encoded = charts._dumps(options)
    split_label = html.escape(split)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Road geometry model comparison</title>
<script src="{charts.ECHARTS_JS}"></script>
<style>
body {{ margin: 0; background: #f3f4f6; color: #111827; font-family: {charts.FONT}; }}
header {{ padding: 20px 24px; background: white; border-bottom: 1px solid #e5e7eb; }}
h1 {{ margin: 0 0 8px; font-size: 24px; }} p {{ line-height: 1.5; max-width: 1000px; }}
select {{ padding: 7px; margin-left: 8px; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 600px), 1fr)); gap: 16px; padding: 16px; }}
.chart {{ height: 480px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; min-width: 0; }}
#error {{ color: #b91c1c; font-weight: bold; }}
</style></head><body><header>
<h1>Road geometry model comparison</h1>
<p>Split: <strong>{split_label}</strong>. Test data only. Predictions are estimates; target units and coordinate CRS are unconfirmed.
Random-row results can share road identifiers across partitions. Grouped results do not prove geographic independence.</p>
<p>Coefficient units differ; magnitudes are not feature importance or causal effects. Location plots use the first geometry vertex, not the segment centroid;
colour is signed log(1 + |residual|), not linear residual magnitude. The histogram includes all test residuals.</p>
<label for="model">Model</label><select id="model"><option value="A">A: length</option><option value="B">B: length + speed</option><option value="C">C: length + speed + lanes</option></select>
<p id="error" role="alert"></p></header>
<main><div id="actual" class="chart"></div><div id="residual" class="chart"></div>
<div id="histogram" class="chart"></div><div id="accuracy" class="chart"></div><div id="r2" class="chart"></div>
<div id="coefficients" class="chart"></div><div id="geography" class="chart"></div></main>
<script>
const error = document.getElementById('error');
if (typeof echarts === 'undefined') {{
  error.textContent = 'Could not load Apache ECharts from the CDN. Check your internet connection.';
}} else {{
  const options = {encoded};
  const ids = ['actual', 'residual', 'histogram', 'accuracy', 'r2', 'coefficients', 'geography'];
  const plots = Object.fromEntries(ids.map(id => [id, echarts.init(document.getElementById(id))]));
  function render() {{
    const selected = document.getElementById('model').value;
    ids.forEach(id => plots[id].setOption(options[selected][id], true));
  }}
  document.getElementById('model').addEventListener('change', render);
  window.addEventListener('resize', () => ids.forEach(id => plots[id].resize()));
  render();
}}
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "model_comparison")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "road_segments.csv")
    parser.add_argument("--split", choices=SPLITS, default="connected_group")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, rows = load_results(args.results_dir, args.csv, args.split)
    options = make_options(result, rows, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dashboard_html(options, args.split), encoding="utf-8")
    print(f"Dashboard written to {args.output}; missing coordinates: "
          f"{options['A']['missing_geometry']} of {len(rows['A'])} test rows.")


if __name__ == "__main__":
    main()
