"""Check filtering, group splits, scoring and reproducibility of model comparisons."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_models", ROOT / "src" / "compare_models.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
CONFIG = comparison.load_config(ROOT / "config" / "model_comparison.json")


class ModelComparisonTests(unittest.TestCase):
    def test_quality_filter_reports_overlapping_reasons(self):
        frame = pd.DataFrame({
            "ROADNAME": ["a", "b", "", "d"], "NID": ["1", "2", "3", "4"],
            "SEGMENT_LENGTH_KM": ["1", "2", "bad", "4"],
            "SPEEDLIMIT": ["100", "0", "50", "90"], "NBRLANES": ["2"] * 4,
            "Shape__Length": ["10", "20", "30", "40"],
        })
        with self.assertRaisesRegex(ValueError, "at least three"):
            comparison.prepare_data(frame)
        frame.loc[2, ["ROADNAME", "SEGMENT_LENGTH_KM"]] = ["c", "3"]
        clean, quality = comparison.prepare_data(frame)
        self.assertEqual(list(clean.source_row), [0, 2, 3])
        self.assertEqual(quality["nonpositive_speed"], 1)
        self.assertEqual(quality["excluded_union"], 1)
        self.assertEqual(quality["eligible"], 3)

    def test_grouped_split_preserves_transitive_identifiers(self):
        frame = pd.DataFrame({
            "ROADNAME": ["a", "a", "b", "c", "d", "e", "f", "g"],
            "NID": ["1", "2", "2", "3", "4", "5", "6", "7"],
        })
        groups = comparison.connected_groups(frame)
        self.assertEqual(groups[0], groups[2])
        parts = comparison.make_splits(frame, CONFIG, True)
        overlap = comparison.overlap_report(frame, parts)
        self.assertTrue(all(v == 0 for group in overlap.values() for v in group.values()))
        self.assertEqual(sum(map(len, parts.values())), len(frame))
        repeat = comparison.make_splits(frame, CONFIG, True)
        for key in parts:
            np.testing.assert_array_equal(parts[key], repeat[key])

    def test_metrics_and_residual_sign(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 2.0, 2.0])
        result = comparison.evaluate(actual, predicted)
        self.assertAlmostEqual(result["mae"], 2 / 3)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2 / 3))
        self.assertAlmostEqual(result["r2"], 0)
        self.assertAlmostEqual(result["residual_p05"], -0.9)
        self.assertIsNone(comparison.evaluate(np.ones(2), np.ones(2))["r2"])

    def test_compare_outputs_common_rows_and_predictions(self):
        rng = np.random.default_rng(23)
        count = 90
        frame = pd.DataFrame({
            "ROADNAME": [f"r{i // 3}" for i in range(count)],
            "NID": [f"id{i}" for i in range(count)],
            "SEGMENT_LENGTH_KM": rng.uniform(0.1, 10, count),
            "SPEEDLIMIT": rng.choice([50, 80, 100], count),
            "NBRLANES": rng.choice([1, 2, 3], count),
        })
        frame["Shape__Length"] = (7 + 4 * frame.SEGMENT_LENGTH_KM
                                  + 0.3 * frame.SPEEDLIMIT - 2 * frame.NBRLANES)
        clean, _ = comparison.prepare_data(frame)
        result, predictions = comparison.compare(clean, CONFIG)
        expected_count = sum(split["counts"]["test"] * 3 for split in result.values())
        self.assertEqual(len(predictions), expected_count)
        for split in result.values():
            self.assertEqual(sum(split["counts"].values()), count)
            for name in ("train", "validation", "test"):
                self.assertEqual(split["models"]["C"]["partitions"][name]["n"], split["counts"][name])
                self.assertLess(split["models"]["C"]["partitions"][name]["rmse"], 1e-10)
        for row in predictions:
            self.assertEqual(row["partition"], "test")
            self.assertAlmostEqual(row["residual"], row["actual"] - row["predicted"])
        json.dumps(result, allow_nan=False)

    def test_invalid_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = dict(CONFIG, train_fraction=0.9, validation_fraction=0.2)
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split fractions"):
                comparison.load_config(path)


if __name__ == "__main__":
    unittest.main()
