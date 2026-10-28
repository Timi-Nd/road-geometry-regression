"""Check the descriptive audit's handling of numerical and geometry data."""

import importlib.util
import json
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "investigate_dataset.py"
spec = importlib.util.spec_from_file_location("investigate_dataset", SCRIPT)
investigate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(investigate)


class DatasetInvestigationTests(unittest.TestCase):
    def test_numeric_quality_distinguishes_blank_text_and_infinity(self):
        result = investigate.numeric_profile(pd.Series(["", "None", "2", "inf", "NaN"]))
        self.assertEqual(
            [result["blank"], result["nonnumeric"], result["nonfinite"], result["finite"]],
            [1, 2, 1, 1],
        )
        self.assertEqual(result["quantiles"]["0.5"], 2)

    def test_geometry_quality_and_bounds(self):
        geometry = pd.Series([
            "",
            "not json",
            '{"paths":[[[1,2],[3,4]]]}',
            '{"paths":[[[1,"bad"]]]}',
            '{"paths":[[]]}',
        ])
        self.assertEqual(
            investigate.geometry_profile(geometry),
            {"blank": 1, "invalid": 3, "valid": 1, "coordinate_bounds": [1, 2, 3, 4]},
        )

    def test_rejects_missing_columns_and_empty_data(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            investigate.profile_dataset(pd.DataFrame({"SPEEDLIMIT": ["100"]}))
        columns = list(investigate.NUMERIC_COLUMNS) + ["ROADNAME", "NID", "geometry_json"]
        with self.assertRaisesRegex(ValueError, "no observations"):
            investigate.profile_dataset(pd.DataFrame(columns=columns))

    def test_profile_handles_missing_values_without_nan_in_json(self):
        frame = pd.DataFrame({
            "KM_FROM": ["0", "2"],
            "KM_TO": ["1", "4"],
            "SEGMENT_LENGTH_KM": ["1", "2"],
            "Shape__Length": ["1600", "3200"],
            "SPEEDLIMIT": ["0", "100"],
            "NBRLANES": ["2", "2"],
            "SURFACE_MATCH_PCT": ["", "100"],
            "GOV_DOMINANT_PCT": ["None", "100"],
            "ROADNAME": ["A", "B"],
            "NID": ["a", "b"],
            "geometry_json": ['{"paths":[[[1,2],[3,4]]]}', ""],
        })
        result = investigate.profile_dataset(frame)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["numeric"]["SURFACE_MATCH_PCT"]["blank"], 1)
        self.assertEqual(result["numeric"]["GOV_DOMINANT_PCT"]["nonnumeric"], 1)
        self.assertEqual(result["fields"]["GOV_DOMINANT_PCT"]["literal_None"], 1)
        self.assertEqual(result["zero_speed_count"], 1)
        self.assertEqual(result["length_ratio_median"], 1.6)
        self.assertIsNone(result["correlations"]["NBRLANES"]["NBRLANES"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
