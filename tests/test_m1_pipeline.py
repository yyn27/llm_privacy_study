"""M1 regression tests: CSV joins, missing values, templates, and record schema."""
from pathlib import Path
import csv
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from synthea_to_records import build_records
from build_dataset import build_patient_rows


class M1PipelineTests(unittest.TestCase):
    def write_table(self, directory, name, headers, rows):
        with (directory / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(rows)

    def make_csv(self, directory):
        self.write_table(directory, "patients.csv",
            ["Id", "FIRST", "LAST", "BIRTHDATE", "SSN", "DRIVERS",
             "PASSPORT", "ADDRESS", "ZIP", "GENDER", "MARITAL"],
            [{"Id": "p-001", "FIRST": "Synthetic", "LAST": "Person",
              "BIRTHDATE": "2000-01-02", "SSN": "999-00-1111",
              "DRIVERS": "", "PASSPORT": "", "ADDRESS": "1 Fictional Street",
              "ZIP": "02100", "GENDER": "F", "MARITAL": ""}])
        self.write_table(directory, "conditions.csv",
            ["PATIENT", "DESCRIPTION", "CODE", "START"],
            [{"PATIENT": "p-001", "DESCRIPTION": "Synthetic Condition",
              "CODE": "X1", "START": "2020-01-01"},
             {"PATIENT": "p-001", "DESCRIPTION": "Synthetic Condition",
              "CODE": "X1", "START": "2020-01-01"}])
        self.write_table(directory, "medications.csv",
            ["PATIENT", "DESCRIPTION", "CODE", "START"],
            [{"PATIENT": "p-001", "DESCRIPTION": "Synthetic Drug",
              "CODE": "M1", "START": "2021-01-01"}])

    def test_joins_and_missing_optional_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.make_csv(directory)
            rows, report = build_records(directory)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["conditions"]), 1)
            self.assertEqual(len(rows[0]["medications"]), 1)
            self.assertEqual(rows[0]["fields"]["passport"], "")
            self.assertEqual(report["field_presence"]["ssn"], 1)
            config = {"probe_fields": ["ssn", "drivers", "passport",
                                       "birthdate", "address", "zip"],
                      "max_conditions_per_patient": 3,
                      "max_medications_per_patient": 3}
            probes, utilities = build_patient_rows(rows[0], "train", config)
            self.assertEqual(len(probes), 4)
            self.assertEqual(len(utilities), 5)
            self.assertTrue(all("Medical record:" not in x["prompt"] for x in probes))
            self.assertTrue(all("Medical record:" in x["prompt"] for x in utilities))
            self.assertTrue(all(x["text"] == x["prompt"] + x["answer"]
                                for x in probes + utilities))

    def test_unmatched_clinical_patient_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.make_csv(directory)
            with (directory / "conditions.csv").open("a", encoding="utf-8") as f:
                f.write("unknown,Unmatched Condition,X2,2022-01-01\n")
            with self.assertRaisesRegex(ValueError, "missing patient IDs"):
                build_records(directory)


if __name__ == "__main__":
    unittest.main()
