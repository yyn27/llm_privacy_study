#!/usr/bin/env python3
"""M1: validate Synthea CSV schema and join patients, conditions, medications."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELD_CATEGORIES = {
    "high_entropy_identifier": ["ssn", "drivers", "passport"],
    "semi_structured": ["birthdate", "address", "zip"],
    "low_entropy_attribute": ["gender", "marital"],
    "clinical_association": ["conditions", "medications"],
}
PATIENT_REQUIRED = {"ID", "FIRST", "LAST", "BIRTHDATE"}
CLINICAL_REQUIRED = {"PATIENT", "DESCRIPTION"}


def read_csv(path: Path, required: set[str]):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        # Synthea patients.csv uses "Id"; normalize all headers rather than
        # assuming the exporter uses uppercase "ID". Reject collisions.
        headers = reader.fieldnames or []
        normalized = [h.strip().upper() for h in headers]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{path.name}: duplicate columns after normalization")
        columns = set(normalized)
        if missing := required - columns:
            raise ValueError(f"{path.name}: missing columns: {sorted(missing)}; "
                             f"actual columns: {headers}")
        for line in reader:
            yield {k.strip().upper(): (v or "").strip()
                   for k, v in line.items() if k is not None}


def sha256_file(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def build_records(csv_dir: Path):
    patients = {}
    duplicates = []
    for p in read_csv(csv_dir / "patients.csv", PATIENT_REQUIRED):
        pid = p["ID"]
        if not pid:
            raise ValueError("patients.csv contains a blank ID")
        if pid in patients:
            duplicates.append(pid)
        patients[pid] = p
    if duplicates:
        raise ValueError(f"Duplicate Synthea patient ID: {duplicates[:3]}")
    if not patients:
        raise ValueError("patients.csv is empty")
    clinical = {}
    join_stats = {}
    for kind in ("conditions", "medications"):
        selected = defaultdict(list)
        unmatched = 0
        for row in read_csv(csv_dir / (kind + ".csv"), CLINICAL_REQUIRED):
            pid, desc = row["PATIENT"], row["DESCRIPTION"]
            if pid not in patients:
                unmatched += 1
                continue
            if desc:
                # Keep dates, useful for deterministic temporal ordering.
                selected[pid].append({"description": desc,
                                      "start": row.get("START", ""),
                                      "code": row.get("CODE", "")})
        for pid, items in selected.items():
            # Stable order independent of Synthea thread and CSV export order.
            items.sort(key=lambda r: (r["start"], r["code"], r["description"]))
            # Avoid duplicate clinical rows with identical start/code/description.
            unique = {}
            for item in items:
                unique[(item["start"], item["code"], item["description"])] = item
            selected[pid] = list(unique.values())
        clinical[kind] = selected
        join_stats[kind] = {"patients_with_rows": len(selected),
                            "unmatched_patient_rows": unmatched,
                            "rows_after_dedup": sum(map(len, selected.values()))}
        if unmatched:
            raise ValueError(f"{kind}.csv has {unmatched} rows for missing patient IDs")
    records = []
    field_presence = Counter()
    for pid in sorted(patients):
        p = patients[pid]
        fields = {
            "ssn": p.get("SSN", ""),
            "drivers": p.get("DRIVERS", ""),
            "passport": p.get("PASSPORT", ""),
            "birthdate": p["BIRTHDATE"],
            "address": p.get("ADDRESS", ""),
            "zip": p.get("ZIP", ""),
            "gender": p.get("GENDER", ""),
            "marital": p.get("MARITAL", ""),
            "city": p.get("CITY", ""),
            "state": p.get("STATE", ""),
        }
        name = " ".join(x for x in [p["FIRST"], p.get("MIDDLE", ""), p["LAST"]] if x)
        if not name:
            raise ValueError(f"Patient {pid} is missing name")
        for name_of_field, value in fields.items():
            if value:
                field_presence[name_of_field] += 1
        records.append({
            "patient_id": pid, "patient_name": name, "fields": fields,
            "conditions": clinical["conditions"].get(pid, []),
            "medications": clinical["medications"].get(pid, []),
        })
    report = {"patient_count": len(records), "field_presence": dict(field_presence),
              "field_categories": FIELD_CATEGORIES, "joins": join_stats,
              "input_sha256": {f: sha256_file(csv_dir / f) for f in
                               ("patients.csv", "conditions.csv", "medications.csv")}}
    return records, report


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", default=str(ROOT / "data/synthea_raw/output/csv"))
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()
    records, report = build_records(Path(args.csv_dir))
    out = Path(args.out)
    write_jsonl(out / "records.jsonl", records)
    (out / "records_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validated {len(records)} patients; wrote {out / 'records.jsonl'}")
    print("Field coverage:", report["field_presence"])
    print("Joins:", report["joins"])


if __name__ == "__main__":
    main()
