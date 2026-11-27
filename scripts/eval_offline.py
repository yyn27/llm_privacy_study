#!/usr/bin/env python3
"""M3 OFFLINE evaluation. Patient outcomes and utility NEVER feed online gate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from utils import append_jsonl, read_jsonl
from audit import answer_nll, greedy_extract, validation_loss

FIELDS={"ssn":"high_entropy_identifier","drivers":"high_entropy_identifier",
        "passport":"high_entropy_identifier","birthdate":"semi_structured",
        "address":"semi_structured","zip":"semi_structured"}


def normalize(x):
    # Provisional: case + repeated whitespace only. Keep punctuation otherwise.
    return " ".join(x.strip().casefold().split())


@torch.inference_mode()
def predict_text(model, tokenizer, prompt, device, max_new_tokens):
    inputs=tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    inputs={k:v.to(device) for k,v in inputs.items()}
    generated=model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated[0,inputs["input_ids"].size(1):],
                            skip_special_tokens=True)


def select_patient_rows(rows, number, seed):
    # Stable patient-ID sample, not first N rows (which would bias fields).
    import random
    patient_ids=sorted({r["patient_id"] for r in rows})
    random.Random(seed).shuffle(patient_ids)
    if number > len(patient_ids):
        raise ValueError(f"Requested {number} patients but only {len(patient_ids)} available")
    selected=set(patient_ids[:number])
    return [r for r in rows if r["patient_id"] in selected]


def field_metrics(model, tok, rows, device, max_length, max_new_tokens):
    by_cat=defaultdict(lambda: {"hits":0,"total":0,"nll_sum":0.0})
    examples=[]
    for row in rows:
        if row["field"] not in FIELDS:
            continue
        target=row["answer"]
        predicted=predict_text(model,tok,row["prompt"],device,max_new_tokens)
        exact=normalize(predicted)==normalize(target)
        loss=answer_nll(model,tok,row["prompt"],[target],device,max_length,1)[0]
        cat=FIELDS[row["field"]]
        for key in (cat,row["field"],"overall"):
            by_cat[key]["hits"]+=int(exact)
            by_cat[key]["total"]+=1
            by_cat[key]["nll_sum"]+=loss
        examples.append({"sample_id":row["sample_id"],"field":row["field"],
                         "category":cat, "exact":bool(exact), "answer_nll":loss})
    summary={k:{"exact_count":v["hits"],"count":v["total"],
                "exact_rate":v["hits"]/v["total"] if v["total"] else None,
                "mean_answer_nll":v["nll_sum"]/v["total"] if v["total"] else None}
             for k,v in by_cat.items()}
    return summary,examples


@torch.inference_mode()
def utility_metrics(model,tok,rows,device,max_new_tokens):
    original=model.training
    model.eval()
    by_type=defaultdict(lambda:{"correct":0,"total":0})
    for row in rows:
        predicted=predict_text(model,tok,row["prompt"],device,max_new_tokens)
        key=row["field"]
        by_type[key]["correct"]+=int(normalize(predicted)==normalize(row["answer"]))
        by_type[key]["total"]+=1
    model.train(original)
    all_correct=sum(x["correct"] for x in by_type.values())
    all_total=sum(x["total"] for x in by_type.values())
    return {"accuracy":all_correct/all_total if all_total else None,
            "correct":all_correct, "total":all_total,
            "by_type":{k:dict(v) for k,v in by_type.items()}}


def evaluate(args):
    start=time.perf_counter()
    cfg=yaml.safe_load((ROOT/args.config).read_text(encoding="utf-8"))
    model_id=yaml.safe_load((ROOT/"configs/models.yaml").read_text(
        encoding="utf-8"))["models"][args.model]["id"]
    out=ROOT/"runs"/args.model/str(args.seed)
    ckpt=out/"adapters"/f"step_{args.step:06d}"/"adapter"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    tok=AutoTokenizer.from_pretrained(model_id, local_files_only=args.local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(
        model_id,dtype=torch.bfloat16,local_files_only=args.local_files_only,
        attn_implementation="eager",trust_remote_code=False)
    model=PeftModel.from_pretrained(base,ckpt).to("cuda:0")
    model.eval()
    device=torch.device("cuda:0")
    train_probe=select_patient_rows(read_jsonl(ROOT/"data/probes_train.jsonl"),
                                    args.train_patients,42)
    test_probe=select_patient_rows(read_jsonl(ROOT/"data/probes_test.jsonl"),
                                   args.unseen_patients,42)
    test_util=select_patient_rows(read_jsonl(ROOT/"data/utility_test.jsonl"),
                                  args.unseen_patients,42)
    train_summary,train_examples=field_metrics(model,tok,train_probe,device,
                                                cfg["train"]["max_length"],
                                                args.max_new_tokens)
    unseen_summary,unseen_examples=field_metrics(model,tok,test_probe,device,
                                                  cfg["train"]["max_length"],
                                                  args.max_new_tokens)
    utility=utility_metrics(model,tok,test_util,device,args.max_new_tokens)
    train_nll=train_summary.get("overall",{}).get("mean_answer_nll")
    unseen_nll=unseen_summary.get("overall",{}).get("mean_answer_nll")
    result={"step":args.step,"seed":args.seed,"model":args.model,
            "field_extract_train":train_summary,"field_extract_unseen":unseen_summary,
            "field_loss_gap_train_minus_unseen": (train_nll-unseen_nll
                if train_nll is not None and unseen_nll is not None else None),
            "utility_qa":utility,
            "evaluated_train_patients":args.train_patients,
            "evaluated_unseen_patients":args.unseen_patients,
            "time_s":time.perf_counter()-start,
            "strictness":"casefold+whitespace normalization; punctuation preserved",
            "gate_eligible":False}
    # No target text or predicted patient string is written to JSONL logs.
    append_jsonl(out/"eval.jsonl", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=["gemma","llama","qwen"],required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--config",default="configs/train_args.yaml")
    p.add_argument("--train-patients",type=int,default=20)
    p.add_argument("--unseen-patients",type=int,default=20)
    p.add_argument("--max-new-tokens",type=int,default=64)
    p.add_argument("--local-files-only",action="store_true")
    args=p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA device is required")
    evaluate(args)


if __name__=="__main__":
    main()
