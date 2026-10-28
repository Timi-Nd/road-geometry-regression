"""Check missing-value eligibility, masking, calibration and estimates."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import impute_length as imputation


CONFIG = imputation.load_config(ROOT / "config" / "imputation.json")


def fixture():
    rng = np.random.default_rng(99)
    count = 100
    length = rng.uniform(0.1, 10, count)
    speed = rng.choice([50, 80, 100], count)
    return pd.DataFrame({
        "ROADNAME": [f"road{i // 2}" for i in range(count)],
        "NID": [f"nid{i}" for i in range(count)],
        "SEGMENT_LENGTH_KM": length,
        "SPEEDLIMIT": speed,
        "Shape__Length": 10 + length * 100 + speed * 2,
    })


class ImputationTests(unittest.TestCase):
    def test_masking_baseline_and_source_integrity(self):
        data = fixture()
        original = data.copy(deep=True)
        quality, validation, final, masked, imputed = imputation.run(data, CONFIG)
        pd.testing.assert_frame_equal(data, original)
        self.assertEqual(quality["missing_target"], 0)
        self.assertEqual(imputed, [])
        self.assertEqual(len(masked), validation["partition_counts"]["test"])
        self.assertLess(validation["regression"]["rmse"], 1e-10)
        self.assertGreater(validation["median"]["rmse"], 1)
        self.assertTrue(all(item["is_artificially_masked"] for item in masked))
        self.assertTrue(all(value == 0 for group in validation["identifier_overlaps"].values()
                            for value in group.values()))
        self.assertEqual(final["training_rows"] + final["calibration_rows"], len(data))

    def test_missing_and_invalid_records_are_separate(self):
        data = fixture()
        data["Shape__Length"] = data["Shape__Length"].astype(object)
        data.loc[0, "Shape__Length"] = np.nan
        data.loc[1, "Shape__Length"] = np.nan
        data.loc[1, "SPEEDLIMIT"] = 0
        data.loc[2, "Shape__Length"] = "not-a-number"
        original = data.copy(deep=True)
        quality, validation, final, masked, imputed = imputation.run(data, CONFIG)
        pd.testing.assert_frame_equal(data, original)
        self.assertEqual(quality["missing_target"], 2)
        self.assertEqual(quality["invalid_nonblank_target"], 1)
        self.assertEqual(final["imputed_rows"], 1)
        self.assertEqual([record["source_row"] for record in imputed], [0, 1])
        self.assertTrue(imputed[0]["is_imputed"])
        self.assertAlmostEqual(imputed[0]["prediction"],
                               10 + 100 * data.loc[0, "SEGMENT_LENGTH_KM"] + 2 * data.loc[0, "SPEEDLIMIT"])
        self.assertFalse(imputed[1]["is_imputed"])
        self.assertEqual(imputed[1]["status"], "not_imputed")
        self.assertEqual(imputed[1]["prediction"], "")
        self.assertNotIn(2, [item["source_row"] for item in imputed])

    def test_holdout_values_do_not_change_fitted_validation_model(self):
        data = fixture()
        _, before, _, rows, _ = imputation.run(data, CONFIG)
        modified = data.copy()
        for row in rows:
            modified.loc[row["source_row"], "Shape__Length"] += 5000
        _, after, _, _, _ = imputation.run(modified, CONFIG)
        self.assertEqual(before["model"], after["model"])
        self.assertEqual(before["median_value"], after["median_value"])
        self.assertEqual(before["interval_radius"], after["interval_radius"])
        self.assertGreater(after["regression"]["rmse"], 4999)

    def test_reproducibility_and_invalid_inputs(self):
        data = fixture()
        first = imputation.run(data, CONFIG)
        second = imputation.run(data, CONFIG)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "missing required"):
            imputation.prepare(data.drop(columns="SPEEDLIMIT"))
        with self.assertRaisesRegex(ValueError, "at least three"):
            imputation.prepare(data.iloc[:2])


if __name__ == "__main__":
    unittest.main()
