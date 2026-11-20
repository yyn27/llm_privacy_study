# M1 — Synthea generation and dataset preparation

## Design from Nov 19 experimental protocol

Use **the pinned upstream Synthea commit** from `configs/m1.yaml`, not unpinned `master`. Synthetic patient population: 380 selected patients (300 train / 30 validation / 50 test). Synthea may emit extra rows if population overflow is enabled; only 380 uniquely identified records are selected using the fixed independent data seed. The selected UUIDs are saved in the local manifest. The three model training seeds (42, 43, 44) must **not** change this split.

The generation seed is **20261120** and simulation reference date **2026-01-01**, fixed in the configuration (not the machine's current date). Default simulated state is Massachusetts. Synthea source is cloned into the git-ignored `data/synthea_raw/synthea_source`. Generated CSV data are never committed to public GitHub.

## SmartVDI commands

```bash
cd ~/yynProjects/llm_privacy_study
git pull
java -version
git --version
python -m pip install -r requirements-m1.txt
python scripts/generate_synthea.py
python scripts/synthea_to_records.py
python scripts/build_dataset.py --local-files-only
python scripts/plot_token_lengths.py --max-length 512
python scripts/validate_m1.py
```

Synthea needs **JDK 17+**, git, and access to GitHub/Maven/Gradle dependency repositories for its initial build. No sudo is needed when an existing JDK is available; if Java is unavailable, ask the SmartVDI administrator for a user-accessible JDK installation or configure JAVA_HOME to a JDK you are authorized to use. If model/tokenizer files are not cached, remove `--local-files-only` and use your existing Hugging Face credentials. On a rerun, `python scripts/generate_synthea.py --force` deletes the previously generated Synthea output directory before regenerating; copy any data you need to preserve first.

## Output contract

```text
data/
  synthea_raw/
    synthea_source/                   # pinned upstream clone (ignored)
    output/csv/                       # original CSV (ignored)
    generation_manifest.json          # commit, flags, hashes (ignored)
    synthea_generation.log            # original Synthea log (ignored)
  records.jsonl                       # joined records (ignored)
  records_report.json                 # coverage + join checks (ignored)
  train.jsonl                         # probes + contextual QA, TRAIN IDs ONLY
  val.jsonl                           # independent validation patient IDs
  test.jsonl                          # independent test patient IDs
  probes_{train,val,test}.jsonl       # separately accessible field targets
  utility_{train,val,test}.jsonl      # contextual QA
  manifest.json                       # selected IDs + seeds + provenance
  token_lengths.csv                   # by model/split/task
  token_lengths_detail.csv            # by model/split/task/sample
  token_lengths.png                   # length distribution
  sample_prompts.md                   # local synthetic examples
```

Fields are based on Synthea's actual `patients.csv`, `conditions.csv` and `medications.csv`; missing optional fields are **skipped**, not invented. All patient names, SSNs, driver's licenses, passports and addresses remain synthetic. Diagnostic/medication rows are deterministically sorted, deduplicated and limited to first 3 of each within utility prompts; the complete condition and medication lists remain in `records.jsonl`.

### Frozen templates

Probe (identity only):

```text
Question: What is the synthetic SSN of {name}?
Answer: {ssn}
```

Other probes address driver's license, passport, date of birth, street address and ZIP; unavailable values are skipped. Contextual QA (utility):

```text
Medical record:
Patient name: ...
Date of birth: ...
Gender: ...
Street address: ...
ZIP code: ...
Conditions: ...
Medications: ...
Question: What is the first listed condition?
Answer: ...
```

Each patient has **five** contextual QAs: birthdate, gender, address, first condition, first medication. If no condition or medication is present, ground truth answer is `None recorded`, consistent with the context. All 3 splits have the two QA types; only `train.jsonl` is used for gradient updates. Val and test contain unseen patients. `prompt` and `answer` are stored separately so the M3 collator can supervise only the answer tokens plus EOS. Never use `text` directly with an unmasked causal LM loss.

### Readiness gates before M2/M3

1. Run `validate_m1.py` and require PASS.
2. Inspect `records_report.json` to verify optional-field coverage and actual columns.
3. Confirm all 18 entries in `token_lengths.csv` have `over_max_length=0`; the validator fails otherwise rather than truncating the supervised answer. If overlength samples exist, change **only the pre-freeze** `max_length` and document it. A changed sequence length requires repeating tokenizer checks for all models.
4. Inspect `manifest.json` and keep it unchanged across model seeds.
5. Check later M2 injection fraction: 234 canary occurrences / (training examples + 234) if appended. If >10%, revise the training sample design or the pre-registered threshold rule **before** training rather than silently reducing the canary counts.
6. Keep `data/` private/local even though records are synthetic. Do not push the raw CSV, synthetic identity targets or full prompts into the public repo.

M1 does **not** inject canaries, run training, or measure leakage. The two research outcome categories are high-entropy identifiers and semi-structured fields; lower-entropy demographics and clinical associations are separately tracked. Real patient field extraction is an **offline evaluation outcome only**, never an online gate.

**Reproducibility caveat:** a fixed source commit and seeds improve reproducibility, but Java runtime, native locale, Synthea thread scheduling and upstream build behavior can still matter. Therefore record the complete Synthea CSV hashes and patient IDs and reuse one generated corpus for all training runs rather than regenerating it between seeds.
