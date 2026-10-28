"""Exercise the imputation command and verify input and output file integrity."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "impute_length.py"


class ImputationCliTests(unittest.TestCase):
    def test_cli_writes_flagged_estimates_and_preserves_input(self):
        rng = np.random.default_rng(55)
        n = 120
        length = rng.uniform(0.1, 10, n)
        speed = rng.choice([50, 80, 100], n)
        frame = pd.DataFrame({
            "ROADNAME": [f"road{i // 2}" for i in range(n)],
            "NID": [f"nid{i}" for i in range(n)],
            "SEGMENT_LENGTH_KM": length,
            "SPEEDLIMIT": speed,
            "Shape__Length": 10 + length * 100 + speed * 2,
        })
        frame.loc[0, "Shape__Length"] = np.nan
        frame.loc[1, "Shape__Length"] = np.nan
        frame.loc[1, "SPEEDLIMIT"] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            frame.to_csv(source, index=False)
            source_bytes = source.read_bytes()
            runs = [root / "out1", root / "out2"]
            for output in runs:
                subprocess.run([sys.executable, str(SCRIPT), "--csv", str(source),
                                "--output-dir", str(output)],
                               cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertEqual(source.read_bytes(), source_bytes)
            for filename in ("imputation_report.json", "masked_validation.csv", "imputed_values.csv"):
                self.assertEqual((runs[0] / filename).read_bytes(), (runs[1] / filename).read_bytes())
            report = json.loads((runs[0] / "imputation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["dataset_sha256"], hashlib.sha256(source_bytes).hexdigest())
            self.assertEqual(report["quality"]["missing_target"], 2)
            self.assertEqual(report["final_imputation_model"]["imputed_rows"], 1)
            with (runs[0] / "imputed_values.csv").open(newline="", encoding="utf-8") as file:
                records = list(csv.DictReader(file))
            self.assertEqual([(row["source_row"], row["is_imputed"]) for row in records],
                             [("0", "True"), ("1", "False")])
            self.assertAlmostEqual(float(records[0]["prediction"]), 10 + length[0] * 100 + speed[0] * 2)
            self.assertEqual(records[1]["prediction"], "")
            self.assertEqual(records[1]["status"], "not_imputed")

    def test_cli_will_not_overwrite_source_as_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "imputation_report.json"
            source.write_text("original data", encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), "--csv", str(source),
                                     "--output-dir", str(root)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not overwrite", result.stderr)
            self.assertEqual(source.read_text(encoding="utf-8"), "original data")


if __name__ == "__main__":
    unittest.main()
