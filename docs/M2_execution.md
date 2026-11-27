# M2 execution — model-specific Canary construction

## Fixed design

M1 is complete: 441 Synthea synthetic patients generated; 380 selected
(300 train, 30 validation, 50 test). M1 train contains 3,187 samples.
**Do not regenerate M1** or edit the existing three data splits.

M2 uses three canary types: random identifier, fictional North American
555-01xx phone number, and random synthetic name–diagnosis association.
For each type and repetition count (1, 5, 20), generate **k=3 distinct
injected canaries**. Thus each model receives **27 distinct injected
canaries, 234 training occurrences**. It also receives **27 distinct
untrained null-gate canaries** and at least **1,000 unique untrained
reference answers per type**. The pool per type is 12,000 unique answers.

The canary prompts retain M1's model-independent `Question:...\nAnswer:`
QA form. For `name_dx`, the random synthetic subject name appears in the
question, and the randomized synthetic diagnosis appears in the answer.
Do **not** interpret the randomized DX code as a realistic clinical
diagnosis. References are candidate answers evaluated under the *same
target prompt* at M3, preventing prompt changes from contaminating the
conditional likelihood ranking.

Phone format uses the conventional fictional 555-0100–555-0199 range;
there are no real contact numbers deliberately introduced. The
three model-specific corpora are **not identical** because tokenizer
length constraints require per-model pools. RQ3 cross-model comparisons
must therefore be described as model-specific matched probes, not exact
identical injected canary strings.

## Run on SmartVDI

```bash
cd ~/yynProjects/llm_privacy_study
git pull
python scripts/validate_m1.py
python scripts/make_canaries.py --model all --local-files-only
python scripts/validate_m2.py
python scripts/report_m2.py
```

Use `--model gemma`, `--model llama`, or `--model qwen` to run separately.
If the tokenizer is not locally cached, omit `--local-files-only`.
The generator rejects existing output to prevent accidental re-sampling.
`--force` explicitly replaces the output for the selected model, and
must **not** be used after M3 begins. If a model failed partway through,
fix the error and rerun **only that model** with `--force`.

All generated files stay in ignored local directories:

```text
canaries/
  gemma/
    injected.jsonl            # 27 unique target canaries
    nullgate.jsonl            # 27 unique untrained controls
    reference.jsonl           # 3000 reference candidate answers
    train_with_canaries.jsonl # 3421 rows; original train + 234 injected appearances
    meta.json                 # answer-token lengths, source hashes, positions
  llama/                      # same file layout, independent tokenizer matching
  qwen/                       # same file layout, independent tokenizer matching
  m2_token_match_table.csv
  proportion_report.txt
```

**M3 training input:** `canaries/{model}/train_with_canaries.jsonl`.
Never use `data/train.jsonl` alone for M3, nor inject null/reference data
into the optimization dataset. M1 `train.jsonl`, `val.jsonl` and
`test.jsonl` remain unchanged.

## Acceptance gates

- For **each** model: exactly 27 unique injected targets, 27 unique
  null targets, and 3,000 reference values (1,000 for each type).
- **Each** injected category/repetition cell: 3 distinct canaries.
- Injected occurrences per model: 3 × 3 × (1 + 5 + 20) = **234**.
- Training sample count: 3,187 + 234 = **3,421**; fraction
  **234 / 3,421 = 6.84%**, below the 10% preset cap.
- Every injected, null and reference answer must be disjoint. A null
  canary must never occur in the training data.
- Within a model and type, injected/null/reference **answer** token
  lengths match exactly. Across different tokenizer families, matching
  lengths is not expected; report their differences in
  `m2_token_match_table.csv`.
- All canary prompts with supervised answer+EOS fit M1's 512-token budget.
- All M1 training samples are preserved, with canary repetitions shuffled
  throughout the training set.
- `python scripts/validate_m2.py` reports `M2 VALIDATION PASS`.

## Measurement boundary / to freeze before M3

M2 **generates data only**; it does not compute exposure, perform prefix
extraction, train an adapter, or claim memory leakage. During M3, calculate
answer-only conditional NLL of each target and 1,000 length-matched
reference candidates under the target's unchanged prompt. Document
empirical exposure rank convention (including whether target itself
participates in rank) in `freeze.md` before looking at results.
Null targets use the same scoring procedure, but are never exposed in
training. Keep any patient-field extraction and utility evaluations
entirely separate from the online canary gate.

If the model's exact tokenizer revision cannot be resolved, compare
cached files/revisions during M3 freeze. Keep `meta.json`,
`m2_token_match_table.csv`, `proportion_report.txt`, and the local
`m1_manifest_sha256` for reproducibility. Canary content should not
be pushed to the public repository.
