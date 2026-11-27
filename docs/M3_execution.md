# M3: LoRA training, online Canary auditing, and independent offline evaluation

## Before running

M2 metadata inspection: all three models have 27 unique injected, 27 unique
null-gate and 3,000 reference candidates; 234 occurrences per model; total
3,421 training rows; injection 6.84%. Tokenizer revisions were unavailable
in the uploaded metadata; record the exact model revision before M4.

This implementation supports **100-step SMOKE TESTS ONLY with the current
`configs/train_args.yaml`**. `configs/freeze.md` is PROVISIONAL.
Do not treat exposure using 64 references as formally comparable to
full exposure using 1,000 references. Do not view early curves and then
secretly tune the pre-registered rules.

### SmartVDI commands

```bash
cd ~/yynProjects/llm_privacy_study
git pull
python scripts/validate_m1.py
python scripts/validate_m2.py
python -m unittest discover -s tests -p 'test_m3_*.py'
```

First smoke test Qwen for 20 steps (one audit). The default M3
`configs/train_args.yaml` audits at steps 20, 40, 60, 80, 100.
The canary audit is costly, so validate the first checkpoint **before**
increasing budget.

```bash
python scripts/train_lora.py --model qwen --seed 42 --max-steps 20 \
  --audit-interval 20 --reference-limit 64 --val-limit 32 \
  --local-files-only
python scripts/eval_offline.py --model qwen --seed 42 --step 20 \
  --train-patients 20 --unseen-patients 20 --local-files-only
```

Then repeat smoke tests for Gemma and Llama, replacing `qwen` with
`gemma` or `llama`. Use a **new run directory** for any changed
configuration; do not reuse `runs/{model}/42` with different step
budgets or audit intervals. For repeatable 100-step M3 smoke tests,
use a clean run directory only if you have archived the 20-step
test output elsewhere and explicitly removed it:

```bash
python scripts/train_lora.py --model qwen --seed 42 --local-files-only
```

The default 100-step M3 smoke emits 5 checkpoints.
All training loss is answer+EOS-only, bf16 LoRA r16/alpha32 and
7 projection targets, batch 1, gradient accumulation 8, LR 2e-4
constant after 20-step warmup.

**Do not run all 3 models simultaneously.** Model and datasets are
separate. M3 does not early-stop on privacy metrics; it logs full
trajectories for later offline replay.

### Per-run outputs

```text
runs/{model}/{seed}/
  config.json
  metrics.jsonl                  # train loss, mixed/utility/probe val loss,
                                 # per-canary exposure and greedy exact
  eval.jsonl                     # OFFLINE ONLY: train/unseen extraction,
                                 # loss gap, utility QA
  adapters/step_000020/
    adapter/                     # PEFT adapter
    trainer_state.pt             # optimizer + random generators (trusted local only)
```

M3 refuses to overwrite an existing run without `--resume`. For a
run interrupted between checkpoint saves, rerun with **the same
parameters** and `--resume`; it resumes from the last saved checkpoint.
If a run fails during an audit after its checkpoint has been written,
inspect the logs carefully: the resume mechanism restores training
state and may not recompute an interrupted audit. Do not claim the run
complete if its metrics.jsonl lacks required checkpoint rows.

### Before M4: formal freeze

The M3 setup is intended for debugging, not hypothesis testing.
Before M4, freeze `configs/freeze.md` and choose, in advance:
full reference pool=1000; full validation (val_limit=0);
max_steps (~30 epoch equivalent for 3421 samples and accumulation 8);
audit frequency; horizon h; leak confirmation rule; normalization;
acceptable utility loss; early-stopping patience/min_delta; null alarm
objective and K; checkpoint rollback rules. The data seed must remain
20261120; seed 42 is calibration, 43/44 untouched for evaluation.
Consider data snapshots/hash of model/tokenizer alongside SHA hashes.

## Limitations to review during smoke

- `strictness` in `eval_offline.py` currently means casefold and
  whitespace only, not final punctuation/date normalization.
- Exposure uses **mean answer-token NLL** and excludes EOS; final paper
  must state the exact scoring convention.
- Exact extraction uses greedy generation and full-string equality,
  not substring match. max_new_tokens is a fixed test budget.
- Offline evaluation has a fixed 20 patient smoke subsample. Formal
  selection should use up to all 300 train and 50 held-out test patients.
- This offline test is not permitted in gating code.
- Peft adapters and trainer state must remain local; no trained
  weights or patient-level outputs belong in the public repo.

## Debug checklist

1. All seven modules and BF16 work (already established by M0).
2. Single 20-step checkpoint exists and its `metrics.jsonl` has exactly
   one line with 54 canary results (27 injected + 27 null).
3. Full-reference mode should use 1000 candidates per canary,
   and report empirical exposure in bits.
4. `eval.jsonl` is created **only after** calling offline evaluator.
5. Zero or low extraction rates are scientific results, not automatic
   proof of safety.
