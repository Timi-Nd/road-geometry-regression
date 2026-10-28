"""Verify dashboard input checks, chart data and generated HTML."""

import importlib.util
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("visualize_models", ROOT / "src" / "visualize_models.py")
visual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visual)


class VisualizationTests(unittest.TestCase):
    def test_saved_experiment_input_hashes_match_checkout(self):
        cases = (
            ("model_comparison", "model_comparison.json", "model_comparison.json"),
            ("imputation", "imputation_report.json", "imputation.json"),
        )
        dataset = ROOT / "data" / "road_segments.csv"
        dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
        for result_dir, report_name, config_name in cases:
            with self.subTest(experiment=result_dir):
                report_path = ROOT / "results" / result_dir / report_name
                report = json.loads(report_path.read_text(encoding="utf-8"))
                config = ROOT / "config" / config_name
                config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
                self.assertEqual(report["dataset_sha256"], dataset_hash)
                self.assertEqual(report["config_sha256"], config_hash)

    def test_coordinate_validation(self):
        self.assertEqual(visual.first_coordinate('{"paths":[[[1,2],[3,4]]]}'), (1, 2))
        for value in ("", "bad", "null", '{"paths":[]}', '{"paths":[[[true,2]]]}',
                      '{"paths":[[[181,2]]]}', '{"paths":[[[1,NaN]]]}'):
            with self.subTest(value=value):
                self.assertIsNone(visual.first_coordinate(value))

    def test_scatter_residual_axis_and_histogram(self):
        x = np.array([3, 5, 6])
        residual = np.array([-3, 2, 0])
        chart = visual.scatter_option(x, residual, "Predicted", "Residual", "title", "subtitle", True)
        self.assertEqual(chart["series"][0]["data"], [[3, -3], [5, 2], [6, 0]])
        self.assertEqual(chart["series"][0]["markLine"]["data"], [{"yAxis": 0}])
        histogram = visual.histogram_option(residual, "subtitle")
        self.assertEqual(sum(histogram["series"][0]["data"]), 3)

    def test_coefficients_and_geography_preserve_values(self):
        model = {"intercept": -1, "coefficients": {"LENGTH": 2, "SPEED": -3}}
        option = visual.coefficient_option(model)
        self.assertEqual([s["data"][0] for s in option["series"]], [-1, 2, -3])
        self.assertEqual(len(option["xAxis"]), 3)
        self.assertTrue(all(axis["min"] <= 0 <= axis["max"] for axis in option["xAxis"]))
        rows = pd.DataFrame({"geometry_json": ['{"paths":[[[1,2],[3,4]]]}', ""],
                             "residual": [-5, 10]})
        geo, missing = visual.geographic_option(rows, "test")
        self.assertEqual(missing, 1)
        self.assertEqual(geo["series"][0]["data"][0][:3], [1, 2, -5])
        self.assertLess(geo["series"][0]["data"][0][3], 0)
        self.assertTrue(geo["xAxis"]["scale"])
        self.assertTrue(geo["yAxis"]["scale"])

    def test_full_experiment_and_dashboard_output(self):
        result, rows = visual.load_results(ROOT / "results" / "model_comparison",
                                           ROOT / "data" / "road_segments.csv", "connected_group")
        options = visual.make_options(result, rows, "connected_group")
        for name in visual.MODELS:
            self.assertEqual(len(options[name]["actual"]["series"][0]["data"]), result["counts"]["test"])
            self.assertEqual(len(options[name]["residual"]["series"][0]["data"]), result["counts"]["test"])
            self.assertEqual(sum(options[name]["histogram"]["series"][0]["data"]), result["counts"]["test"])
            self.assertEqual(len(options[name]["geography"]["series"][0]["data"])
                             + options[name]["missing_geometry"], result["counts"]["test"])
        self.assertEqual(options["A"]["accuracy"]["series"][1]["data"][1],
                         result["models"]["B"]["partitions"]["test"]["rmse"])
        self.assertEqual(options["A"]["r2"]["series"][0]["data"][2],
                         result["models"]["C"]["partitions"]["test"]["r2"])
        page = visual.dashboard_html(options, "connected_group")
        self.assertIn("model.addEventListener", page.replace("document.getElementById('model')", "model"))
        self.assertIn("Could not load Apache ECharts", page)
        self.assertNotIn('"function(', page)

    def test_dataset_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "data.csv"
            fake.write_text("test", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                visual.load_results(ROOT / "results" / "model_comparison", fake, "connected_group")

    def test_tampered_predictions_fail(self):
        source = ROOT / "results" / "model_comparison"
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            shutil.copyfile(source / "model_comparison.json", dest / "model_comparison.json")
            predictions = pd.read_csv(source / "model_predictions.csv", dtype={"ROADNAME": str, "NID": str})
            selected = predictions.index[(predictions["split"] == "connected_group")
                                         & (predictions["model"] == "A")][0]
            predictions.loc[selected, "predicted"] += 1
            predictions.to_csv(dest / "model_predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "predictions do not match"):
                visual.load_results(dest, ROOT / "data" / "road_segments.csv", "connected_group")


if __name__ == "__main__":
    unittest.main()
