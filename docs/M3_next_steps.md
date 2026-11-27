# M3 post-smoke continuation (2026-11-27)

The three initial 20-step smoke tests succeeded. Their existing
`runs/{model}/42/` outputs MUST NOT be overwritten.

## New M3 100-step smoke protocol

These additional checks are engineering validation, NOT the M4 formal memory
curve. After reviewing and freezing `configs/freeze.md`, formal M4 must use
full 1000-reference exposure with fixed audit interval and full held-out
validation, and cannot compare absolute exposure values with this 64-reference
smoke.

```bash
cd ~/yynProjects/llm_privacy_study
git pull
python -m compileall -q scripts configs tests
python -m unittest discover -s tests -p 'test_m3_*.py'
python scripts/validate_m2.py
```

Run **one model at a time**. All use the unchanged M2 canaries, seed 42,
same hyperparameters and 100 optimizer steps.

```bash
python scripts/train_lora.py --model qwen --seed 42 \
    --run-tag smoke100 --max-steps 100 --audit-interval 20 \
    --reference-limit 64 --val-limit 0 --audit-step-zero --local-files-only
python scripts/eval_offline.py --model qwen --seed 42 --run-tag smoke100 \
    --step 100 --train-patients 20 --unseen-patients 20 --local-files-only
```

Repeat both commands with `gemma` and `llama`.
**Step 0 audit** scores the untrained base+fresh LoRA adapter BEFORE any
optimizer updates; its Canary `actual_seen_so_far` values are all zero.
Every subsequent checkpoint records the actual number of training
microbatches containing each injected Canary, replayed from the exact
deterministic shuffled order. A nullgate count must remain zero.

Validation on **all 319 samples** is enabled by `--val-limit 0`.
Use **64 reference strings** during these engineering tests; formal
exposure needs 1000 and its values are not numerically comparable.
The `--run-tag smoke100` route stores all output separately at
`runs/{model}/42/smoke100` and preserves earlier 20-step runs.

The base model's Qwen/Llama/Gemma tokenizer outputs are now decoded
with `clean_up_tokenization_spaces=False`. This suppresses the
Llama BPE warning and makes string matching behavior explicit.
It does not require retraining old adapters.

## Validate complete runs

```bash
python scripts/validate_m3.py --run-tag smoke100 --seed 42 \
    --models qwen,gemma,llama --max-steps 100 --audit-interval 20 \
    --require-step-zero
```

Expected audit steps **0, 20, 40, 60, 80, 100**, each with 54 targets
(27 injected and 27 null). Checkpoints are saved for steps 20..100.
The metric JSONL now additionally stores `microstep` and per-canary
`actual_seen_so_far` for interpreting exposure versus actual contact.

Inspect `runs/{model}/42/smoke100/metrics.jsonl`, `eval.jsonl`, and
`config.json` to verify readiness. Do not classify a temporary
smoke-test rise in exposure as leakage.

## IMPORTANT: before formal M4

The initial 20-step tests were only ~0.047 epochs and cannot identify
memorization trajectories. The 100-step check is ~0.234 epochs, still
shorter than one epoch. M4 requires substantially longer training,
predeclared decision rules, fixed full 1000-reference exposures, complete
validation, and **offline patient extraction at all checkpoints**.

Model revision in recent M3 config is still `unavailable`; its
tokenizer fingerprint is saved. For M4, record the resolved Hugging Face
model snapshot SHA or compare against M0 `models_table.csv` from the
same machine. Record environment software versions alongside hashes.

The original research plan suggests auditing **at least five times per
epoch**. With 3421 examples and gradient accumulation 8, one epoch is
roughly 428 optimizer steps, so N=~80 or below would satisfy this,
and ~30 epochs would be ~12,830 optimizer steps. These are calculations,
NOT approved formal frozen choices. Their compute/storage cost must be
measured before committing. For example, 12,830/80 is ~160 audits per
run; 54 canaries x 1001 answer scores makes long experiments expensive.
Pilot benchmark and set a realistic budget before freezing.
