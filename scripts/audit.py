"""M3 online audit: injected/null canary exposure + greedy prefix extraction only."""
import math
import time
import torch
from utils import encode_qa, tensor_batch


@torch.inference_mode()
def answer_nll(model, tokenizer, prompt, answers, device, max_length=512, batch_size=8):
    """Per-answer mean token NLL. Excludes EOS, prompt, and padding."""
    results = []
    pad = tokenizer.pad_token_id
    for start in range(0, len(answers), batch_size):
        enc = [encode_qa(tokenizer, prompt, a, max_length, include_eos=False)
               for a in answers[start:start + batch_size]]
        batch = tensor_batch(enc, pad, device)
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = batch["labels"][:, 1:]
        valid = shift_labels != -100
        safe_targets = shift_labels.masked_fill(~valid, 0)
        lp = torch.log_softmax(shift_logits, -1).gather(
            -1, safe_targets.unsqueeze(-1)).squeeze(-1)
        counts = valid.sum(dim=-1)
        if (counts == 0).any():
            raise ValueError("No supervised answer tokens after causal shift")
        per_row = -(lp * valid).sum(dim=-1) / counts
        results.extend(per_row.cpu().tolist())
        del logits, shift_logits, shift_labels, lp
    return results


@torch.inference_mode()
def greedy_extract(model, tokenizer, prompt, answer, device, max_new_tokens=48):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k,v in encoded.items()}
    target_token_count = len(tokenizer(answer, add_special_tokens=False)["input_ids"])
    budget = min(max_new_tokens, max(1, target_token_count + 4))
    output = model.generate(
        **inputs, max_new_tokens=budget, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    # Full-answer extraction; no substring acceptance.
    normalized = lambda x: " ".join(x.strip().split()).casefold()
    return normalized(text) == normalized(answer)


def audit_canaries(model, tokenizer, injected, nulls, references,
                   max_length, device, ref_limit=1000, batch_size=8,
                   max_new_tokens=48):
    if ref_limit < 2:
        raise ValueError("Need at least 2 reference candidates")
    by_kind = {}
    for r in references:
        by_kind.setdefault(r["category"], []).append(r["answer"])
    start = time.perf_counter()
    original_training = model.training
    model.eval()
    scores = []
    try:
        # DO NOT inspect patient fields here. This routine is online gate-only.
        for group, examples in (("injected", injected), ("nullgate", nulls)):
            for target in examples:
                refs = by_kind[target["category"]][:ref_limit]
                if len(refs) != ref_limit:
                    raise ValueError(f"Missing reference answers: {target['category']}")
                # Every reference is scored under the EXACT target prompt.
                losses = answer_nll(model, tokenizer, target["prompt"],
                                    [target["answer"]]+refs, device,
                                    max_length, batch_size)
                rank = 1 + sum(nll <= losses[0] for nll in losses[1:])
                exposure = math.log2((len(refs)+1)/rank)
                exact = greedy_extract(model, tokenizer, target["prompt"],
                                       target["answer"], device, max_new_tokens)
                scores.append({"membership": group, "canary_id": target["canary_id"],
                               "category": target["category"],
                               "repeat": target["repeat"],
                               "reference_count": len(refs), "rank": rank,
                               "exposure": exposure, "prefix_exact": bool(exact)})
    finally:
        model.train(original_training)
    from collections import defaultdict
    summary = defaultdict(list)
    for row in scores:
        summary[(row["membership"], row["category"], row["repeat"])].append(row)
    aggregate = {}
    for key, items in summary.items():
        group, category, repeats = key
        aggregate.setdefault(group, {}).setdefault(category, {})[str(repeats)] = {
            "mean_exposure": sum(x["exposure"] for x in items)/len(items),
            "max_exposure": max(x["exposure"] for x in items),
            "prefix_exact_count": sum(x["prefix_exact"] for x in items),
            "count": len(items)}
    return {"canary_scores": scores,
            "canary_summary": aggregate,
            "audit_seconds": time.perf_counter()-start,
            "reference_limit": ref_limit}


@torch.inference_mode()
def validation_loss(model, tokenizer, examples, max_length, device):
    """Answer-token weighted mixed and Utility-only loss, independent of gate."""
    import torch.nn.functional as F
    prior = model.training
    model.eval()
    groups = {"mixed": [0.0, 0], "utility": [0.0, 0], "probe": [0.0, 0]}
    try:
        for row in examples:
            x = tensor_batch([encode_qa(tokenizer, row["prompt"], row["answer"],
                                        max_length)], tokenizer.pad_token_id, device)
            logits = model(input_ids=x["input_ids"],
                           attention_mask=x["attention_mask"]).logits[:, :-1].float()
            targets = x["labels"][:, 1:]
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       targets.reshape(-1), ignore_index=-100,
                                       reduction="sum").item()
            tokens = int((targets != -100).sum().item())
            for name in ("mixed", row["task_type"]):
                groups[name][0] += loss_sum
                groups[name][1] += tokens
    finally:
        model.train(prior)
    return {k+"_loss": (v[0]/v[1] if v[1] else None) for k,v in groups.items()}
