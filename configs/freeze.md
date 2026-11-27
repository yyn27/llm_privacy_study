# M3 experiment freeze — PROVISIONAL (not final)

**Do not claim this is a frozen experiment until the researcher reviews and
replaces provisional values and commits it before viewing M4 curves.**
M3 100-step runs are smoke tests, not calibrated M4 trajectories.

- Synthea pinned commit: `bd8a52ba0a54ad8c6a1004b74787a14d8c25dbcc`
- Data seed: `20261120`; M1: 300/30/50 patients; M2: 27 injected
  canaries per model (234 appearances), 27 null canaries and 3,000
  reference values, separately token-matched per model.
- Models: Gemma 3 1B IT, Llama 3.2 1B Instruct, Qwen2.5 1.5B Instruct.
- LoRA: r=16, alpha=32, dropout=.05, bias=none,
  q/k/v/o/gate/up/down projections. BF16, not QLoRA.
- One shared plain-text QA prompt, answer+EOS-only supervision.
  Never supervise prompt tokens. Canary rank scoring excludes EOS.
- Checkpoint audit uses **only canary exposure and greedy prefix exact**.
  Train loss and validation loss may be logged, but patient fields,
  clinical utility and stronger attacks must NEVER enter gate rules.
- Exposure empirical proxy: `log2((R+1)/rank)`, where rank =
  `1 + number of reference candidates with NLL <= target NLL`.
  Ties are conservatively counted against targets, per-target prompt
  is unchanged for ALL candidate answers. R=1000 in formal runs;
  R=64 only in smoke tests (these scores are not comparable).
- Extraction: fixed `Question: ...\\nAnswer:` prompt, greedy generation,
  strict target string match after stripping whitespace and EOS.
  Each scored target gets an independent prompt; generated answer
  is compared against full target, not only a prefix substring.
- Dataset splits and token budget: max_length 512.
- Training schedule: constant LR 2e-4 after 20 warmup optimizer steps,
  train batch 1, accumulation 8, same for every model. Seeds 42
  calibration, 43/44 evaluation; dataset and canary seed stays fixed.
- Validation early stopping: primary mixed Probe+Utility answer-only loss,
  also log Utility-only validation loss.
- **TO FREEZE before M4**: full audit interval N (recommend >=5 per
  epoch); max_steps spanning ~30 epochs; leakage-event confirmation at
  subsequent checkpoint; prediction horizon h; validation patience and
  min_delta; return-to-best-checkpoint rule; fixed 3/5/10 epoch comparison;
  gate threshold/null false-alarm objective and consecutive K rule;
  acceptable utility decline delta_u; deterministic normalization
  rules and field endpoint inclusion/exclusion. These have intentionally
  not been invented as final research decisions.
- True leakage event: individual (train patient, field), for high-entropy
  identifiers and semi-structured categories, must be exact and confirmed
  again at the following audit. At least one confirmed field per patient
  in primary event. Unseen controls must be separated.
- Offline evaluation: no patient information can affect online stopping.
- Replay of policies cannot simulate adaptive audit spacing unless
  intermediate checkpoints exist. This system provides empirical audit,
  **not** differential privacy or legal compliance.
