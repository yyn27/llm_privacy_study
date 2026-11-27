# M2-v2 control repair and 1000-reference audit benchmark

## Why repair?

M2-v1 QA prompts included model ID and membership in the canary subject,
e.g. `CanarySubject_phone_qwen_injected_007`. This is a confound for
injected/null differences. **Do not modify or overwrite v1 outputs**:
keep the already completed 20- and 100-step smoke results as engineering
tests only. M2-v2 gives each canary a deterministic opaque subject
(`SyntheticPerson_<hex>`), with no model, experimental arm or repeat level
exposed in the prompt. Membership, repetition and model remain metadata only.
The QA template, lengths, 27+27 targets, 3000 references and 234
injections per model stay unchanged.

## Phase A — generate and validate M2-v2

```bash
cd ~/yynProjects/llm_privacy_study
git pull
python -m compileall -q scripts tests
python -m unittest discover -s tests -p 'test_m2_v2.py'
python scripts/make_canaries_v2.py --model all --local-files-only
python scripts/validate_m2_v2.py
python scripts/report_m2.py --dir canaries_v2
```

Generated data are stored separately in ignored `canaries_v2/{model}`.
**Do not use `--force`** unless a first generation failed and the
existing target model directory is known incomplete. Keep M2-v1 exactly
as it was.

The validator rejects prompts containing 'injected', 'nullgate',
model name, 'repeat' or 'canarysubject'; checks unique subjects, prompts,
answers, repetition counts, positions and M1 hashes. The M2-v2 metadata
records a tokenizer fingerprint.

## Phase B — benchmark with old Step 100 adapter: NO TRAINING

The existing `runs/{model}/42/smoke100` adapter was trained on M2-v1,
**not M2-v2**. It is still valid to benchmark GPU memory and execution
time on M2-v2-shaped inputs, but exposure and extraction scores from
this combination are **not scientific results** and must not appear
in the study's M4 plots.

For the first benchmark, only score **6 canaries total** (one injected
and one null per category) against 1000 reference candidates each:

```bash
python scripts/benchmark_1000refs.py --model qwen --canary-version v2 \
  --targets-per-category 1 --batch-size 8 --local-files-only
```

If no out-of-memory error, run the complete set of **54 canaries**:

```bash
python scripts/benchmark_1000refs.py --model qwen --canary-version v2 \
  --batch-size 8 --local-files-only
```

Then repeat first with `gemma` and `llama` (one model at a time).
If GPU OOM occurs, retry **with --batch-size 4**, recording the changed
batch size, rather than lowering the reference count.

Optional matching-control benchmark of historic data: set
`--canary-version v1` for the old M2-v1 dataset and old adapter.
This is still a timing test; it should never substitute for M4-v2.

Results are saved to `outputs/benchmarks/{model}_{version}_{N}targets_1000refs.json`.
They include total seconds, seconds per target, GPU peak allocated and
reserved GiB and a linear estimate for a complete 54-target audit.
Full audit includes greedy generation. The script does not train, edit
or save any adapter and does not write sensitive target texts.
Existing results are protected against overwrites.

The full audit consumes **54 × 1001 = 54,054 candidate-answer NLLs**
per model (plus 54 greedy generations), so even a technically successful
smoke can incur substantial runtime. Benchmark all three before freezing
M4 audit interval or long-run budget. The linear extrapolation from
6 targets is just a rough indication; full timings are definitive.

## Phase C — formal work, NOT part of this benchmark

Future LoRA training on repaired data MUST explicitly use
`python scripts/train_lora.py --canary-version v2 --run-tag <new-tag> ...`.
Do not resume a v1 model using v2 data. A new adapter must be trained
from the base model on v2 data before interpreting exposure/precision.

For the eventual M4 run:
- freeze exact model snapshots and tokenizer file fingerprints,
  the 1000-reference audit protocol and interval, event confirmation
  rule and online-stop policy before looking at the curves
- use all 319 validation samples
- compare at least Step 0 and subsequent checkpoints using the same
  v2 target/reference pool
- calculate full patient-field and utility outcomes separately,
  offline and never inside the gate

## What to send back for evaluation

1. Terminal output of `validate_m2_v2.py`.
2. The three `canaries_v2/{model}/meta.json` files.
3. Three FULL `outputs/benchmarks/*_v2_54targets_1000refs.json` reports.
4. For any OOM, error trace and GPU memory summary.
