#!/usr/bin/env python3
"""Reproducibly run pinned upstream Synthea (requires git, Java JDK >=17)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def command(cmd, cwd=None, log=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode:
        print(proc.stdout[-6000:], file=sys.stderr)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {cmd[0]}")
    return proc.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/m1.yaml"))
    parser.add_argument("--raw-dir", default=str(ROOT / "data/synthea_raw"))
    parser.add_argument("--force", action="store_true",
                        help="Delete existing generated output only; preserve upstream clone")
    args = parser.parse_args()
    c = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["synthea"]
    raw = Path(args.raw_dir).resolve()
    clone = raw / "synthea_source"
    output = raw / "output"
    if shutil.which("git") is None or shutil.which("java") is None:
        raise RuntimeError("Install git and JDK 17+ in your user environment first.")
    # Gradle requires a Java compiler, not just the java runtime.
    # Fail early with actionable advice instead of downloading/building first.
    javac = shutil.which("javac")
    java_home = os.environ.get("JAVA_HOME", "")
    if not javac:
        raise RuntimeError(
            "Java compiler (javac) not found. A full JDK 17+ is required; "
            "the Java runtime alone is insufficient. Install a user-local "
            "JDK and set JAVA_HOME and PATH before rerunning."
        )
    if java_home:
        home = Path(java_home).expanduser().resolve()
        if not (home / "bin" / "java").is_file() or not (home / "bin" / "javac").is_file():
            raise RuntimeError(
                f"JAVA_HOME={java_home} does not contain both bin/java and "
                "bin/javac. Point JAVA_HOME to a complete JDK directory."
            )
    java_version = command(["java", "-version"])
    compiler_version = command(["javac", "-version"])
    if not clone.exists():
        raw.mkdir(parents=True, exist_ok=True)
        command(["git", "clone", c["repository"], str(clone)])
    if not (clone / ".git").exists():
        raise RuntimeError(f"Not a Git repository: {clone}")
    expected = c["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("Pin synthea.commit to a full 40-character SHA")
    # A dedicated clone only; avoid modifying an unrelated checkout.
    command(["git", "fetch", "origin", expected], cwd=clone)
    command(["git", "checkout", "--detach", expected], cwd=clone)
    actual = command(["git", "rev-parse", "HEAD"], cwd=clone)
    if actual != expected:
        raise RuntimeError(f"Synthea checkout mismatch: {actual}")
    csv_dir = output / "csv"
    if csv_dir.exists():
        if not args.force:
            raise FileExistsError(f"{csv_dir} exists. To regenerate, pass --force.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    # Explicitly freeze generation and exporter options. With --config overrides,
    # CSV is enabled and FHIR is disabled. An absolute output path prevents mixing.
    run = [
        "./run_synthea", "-s", str(c["seed"]),
        "-cs", str(c["clinician_seed"]), "-r", str(c["reference_date"]),
        "-p", str(c["requested_population"]),
        "--exporter.csv.export=true",
        "--exporter.fhir.export=false",
        "--exporter.csv.append_mode=false",
        "--exporter.csv.folder_per_run=false",
        "--exporter.baseDirectory=" + str(output) + "/",
        "--generate.thread_pool_size=" + str(c["threads"]),
        c["state"],
    ]
    log = raw / "synthea_generation.log"
    try:
        command(run, cwd=clone, log=log)
    except Exception:
        print(f"Full Synthea output: {log}", file=sys.stderr)
        raise
    required = ["patients.csv", "conditions.csv", "medications.csv"]
    missing = [x for x in required if not (csv_dir / x).exists()]
    if missing:
        raise RuntimeError(f"Synthea missing CSV files: {missing}")
    checksums = {}
    for f in sorted(csv_dir.glob("*.csv")):
        checksums[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    provenance = {"synthea_repo": c["repository"], "synthea_commit": actual,
                  "synthea_seed": c["seed"], "clinician_seed": c["clinician_seed"],
                  "reference_date": c["reference_date"], "state": c["state"],
                  "requested_population": c["requested_population"],
                  "threads": c["threads"], "command": run,
                  "csv_sha256": checksums, "csv_dir": str(csv_dir),
                  "java_version": java_version[:300],
                  "javac_version": compiler_version[:300],
                  "java_home": java_home}
    (raw / "generation_manifest.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print("Generation succeeded:", csv_dir)


if __name__ == "__main__":
    main()
