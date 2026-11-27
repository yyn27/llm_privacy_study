# LLM Privacy Study — Canary-gated LoRA fine-tuning

Research implementation for studying the relationship between synthetic canary memorization and extractability of fields from **Synthea synthetic patient records**. Results from this project cannot be interpreted as a formal privacy guarantee. The M0 scripts are **environment tests, not experimental evidence**.

## Milestones

- **M0 (implemented):** SmartVDI environment inspection, base-model parameter counts, seven LoRA module checks, 10-step BF16 smoke training.
- **M1 (implemented; requires SmartVDI run):** pinned Synthea CSV generation, 300/30/50 patient-level splits, record/probe/utility text processing, tokenizer lengths and validation. See [M1 execution guide](docs/M1_execution.md).\n- **M2 (implemented; requires SmartVDI run):** tokenizer-matched canary construction, 234 training occurrences per model, null-gate controls and empirical-exposure reference sets. See [M2 execution guide](docs/M2_execution.md).
- M3–M11 (not yet implemented): checkpoint logging, post-hoc evaluation, calibration, policy replay, stronger attacks and reproducibility.

## M0 quick start (SmartVDI / VS Code terminal)

Use the Python environment in which you successfully loaded all three Hugging Face models. CUDA, PyTorch and the GPU driver must already be configured; do not reinstall a working GPU-enabled PyTorch unnecessarily.

```bash
git clone https://github.com/yyn27/llm_privacy_study.git
cd llm_privacy_study
python -m pip install -r requirements-m0.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
nvidia-smi
```

If you already cloned the repository, run `git pull` instead. For gated Llama / Gemma models, accept their model licenses on Hugging Face and authenticate **on the SmartVDI machine** if necessary with `hf auth login`. Never put tokens or cache contents in this repository.

First check one model without training:

```bash
python env_check.py --models qwen --smoke-steps 0
```

Then run a 10-step BF16 LoRA smoke test on **each** model, sequentially (no simultaneous model residency):

```bash
python env_check.py --models gemma,llama,qwen --smoke-steps 10
```

For models already cached and offline VDI sessions:

```bash
python env_check.py --models gemma,llama,qwen --smoke-steps 10 --local-files-only
```

An optional longer 20-step confirmation, as mentioned in the experiment plan's script table, is:

```bash
python env_check.py --models gemma,llama,qwen --smoke-steps 20
```

If a model fails, isolate it: `python env_check.py --models gemma --smoke-steps 10`. Errors are recorded rather than silently ignored; execution exits nonzero if any selected model fails.

### Generated artifacts

All results go into the local **git-ignored** `outputs/m0/` directory:

| File | What to check |
| --- | --- |
| `env.txt` | Python/PyTorch/CUDA versions, full installed-package list, `nvidia-smi`, disk availability |
| `env.json` | Machine-readable system/GPU information |
| `models_table.csv` | Parameter counts, trainable proportions, module checks, revisions, smoke status |
| `models_table.json` | Table plus individual smoke-test losses |

The parameter count is based on unique parameter objects (tied weights counted once). "Non-embedding" excludes **both input and output embedding modules by identity**, even when weights are shared. The percentage of trainable LoRA parameters is calculated relative to **base model** parameter counts, explicitly labeled in the CSV. All seven target names must occur in each model before a successful status can be reported.

The M0 smoke test uses eight small synthetic **non-patient** QA pairs, a shared plain-text prefix, and **loss only over answer and EOS tokens**. This is a compatibility test; its learning rate, batch size and sequence length are **not** frozen as formal M3 hyperparameters. M0 does not save or upload adapters or Hugging Face credentials.

### Manual M0 evidence checklist

- Screenshot `nvidia-smi` showing H200 MIG allocation and actual memory (~71 GiB if configured as planned).
- Check SmartVDI **session timeout**, network access and storage quota with the platform administrator/UI; these are not reliably discoverable by Python alone.
- Verify all three rows in `models_table.csv` have `status=ok`, seven matched targets and 10/20 steps.
- Keep the local `env.txt`, `models_table.csv` and screenshots for the final reproduction package; do not publicly commit machine-identifying paths without review.
- Record any cached-model revision that is unavailable in offline mode before freezing M3 runs.

### Scope and methodological safeguards

This implementation follows the **Nov 19, 2026 three-model experiment procedure**. Model IDs: `google/gemma-3-1b-it`, `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`. Common LoRA: `r=16`, `alpha=32`, `dropout=0.05`, all seven attention/MLP projection modules.

For later milestones, calibration seed **42** and evaluation seeds **43/44** must remain separate. Online gates use **canary exposure and canary prefix extraction only**. Patient-field extraction and utility are strictly offline outcomes; no n-gram gate or LoRA rank ablation is planned. Before M3, freeze the intended leak events, checkpoint interval, training hyperparameters, utility tolerance, and false-positive criteria.

## Security

The repository is public. Do **not** commit Hugging Face access tokens, model weights, patient data (including raw exports), local environment files, adapters or audit logs containing generated targets. Such outputs are git-ignored by default.
