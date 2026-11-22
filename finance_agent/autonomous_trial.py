"""One complete SFT epoch and before/after unhinted task evaluation."""
import argparse
import gc
import json
import math
from pathlib import Path
from random import Random

from peft import LoraConfig, PeftModel, get_peft_model
import torch
from transformers import AutoTokenizer

from .agent import MODEL_ID, REVISION
from .autonomous import execute, grade
from .expand_sft import read_jsonl
from .filings import RUNTIME
from .model_service import encode_prompt, generate_text
from .prepare_autonomous import DATASET
from .prepare_sft import sha256
from .sft_pilot import SNAPSHOT, TARGET_MODULES, dump, load_base, probe_logits, train

OUTPUT = RUNTIME / "sft-autonomous"


def training_batches(tokenizer, examples, seed=42):
    shuffled = examples.copy()
    Random(seed).shuffle(shuffled)
    batch = []
    for row in shuffled:
        prefix = encode_prompt(tokenizer, row["prompt"]).input_ids[0].tolist()
        target = tokenizer(row["completion"][0]["content"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        if len(prefix) + len(target) > 3072:
            raise ValueError(f"{row['case_id']}: sample exceeds context budget")
        batch.append((row["case_id"], row["turn"], prefix + target, target))
        if len(batch) == 4:
            yield batch
            batch = []
    if batch:
        yield batch


def summarize(rows):
    summary = {}
    for category in ("direct", "bridge", "stop", "missing"):
        subset = [row for row in rows if row["case"]["category"] == category]
        summary[category] = {
            "cases": len(subset), "passed": sum(row["checks"]["passed"] for row in subset),
            "terminal_correct": sum(row["checks"]["terminal_correct"] for row in subset),
            "second_search": sum(row["checks"]["second_search"] for row in subset),
            "second_hop_hit": sum(row["checks"]["second_hop_hit"] for row in subset),
            "unnecessary_search": sum(row["checks"]["unnecessary_search"] for row in subset),
            "premature_stop": sum(row["checks"]["premature_stop"] for row in subset),
            "invalid_actions": sum(any(step["action"] == "invalid" for step in row["trace"]) for row in subset),
        }
    return summary


def evaluate(model, tokenizer, cases, stage, output):
    model.eval()
    rows = []
    for i, case in enumerate(cases, 1):
        row = execute(case, lambda prompt: generate_text(model, tokenizer, prompt, "cuda", max_new_tokens=96))
        rows.append(row)
        print(f"{stage} {i}/{len(cases)} {case['category']} {case['id']} passed={row['checks']['passed']} "
              f"searches={row['search_count']} terminal={row['terminal_status']}", flush=True)
    summary = summarize(rows)
    payload = {"next_step_hints": False, "rule_fallback": False, "summary": summary, "rows": rows}
    dump(output / f"{stage}.json", payload)
    return payload


def refresh_scores(output=OUTPUT):
    """Recompute checks from frozen trajectories using the same scoring rules."""
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    for stage in ("before", "after"):
        payload = json.loads((output / f"{stage}.json").read_text(encoding="utf-8"))
        payload["rows"] = [grade(row) for row in payload["rows"]]
        payload["summary"] = summarize(payload["rows"])
        dump(output / f"{stage}.json", payload)
        report[stage] = payload["summary"]
    report["scoring"] = "Require explicit terminal decision; missing evidence requires an empty lookup of the requested target report."
    dump(output / "report.json", report)
    return report


def run(output=OUTPUT, parent=None):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve existing results, use a fresh run directory: {output}")
    examples = read_jsonl(DATASET / "train-actions.jsonl")
    cases = read_jsonl(DATASET / "eval-episodes.jsonl")
    if set(row["company_code"] for row in examples) & set(row["company_code"] for row in cases):
        raise ValueError("Issuer split leaked between train and evaluation")
    parent_report = None
    if parent is not None:
        parent_report = json.loads((parent / "report.json").read_text(encoding="utf-8"))
        for key, source in (
            ("train_sha256", DATASET / "train-actions.jsonl"),
            ("eval_sha256", DATASET / "eval-episodes.jsonl"),
            ("adapter_sha256", parent / "adapter/adapter_model.safetensors"),
        ):
            if parent_report[key] != sha256(source):
                raise ValueError(f"Parent run {key} does not match the current input")
        if parent_report["revision"] != REVISION:
            raise ValueError("Parent adapter belongs to another base-model revision")
    total_epochs = (parent_report.get("total_epochs", parent_report["epochs"]) if parent_report else 0) + 1
    seed = 41 + total_epochs
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    # Materialize and validate before committing GPU time.
    batches = list(training_batches(tokenizer, examples, seed))
    base = load_base(SNAPSHOT)
    if parent is None:
        before = evaluate(base, tokenizer, cases, "before", output)
        model = get_peft_model(base, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0, target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM", revision=REVISION,
        ))
    else:
        before = json.loads((parent / "after.json").read_text(encoding="utf-8"))
        if before["rows"] and [row["case"] for row in before["rows"]] != cases:
            raise ValueError("Parent evaluation does not match the current cases")
        before["rows"] = [grade(row) for row in before["rows"]]
        before["summary"] = summarize(before["rows"])
        dump(output / "before.json", before)
        model = PeftModel.from_pretrained(base, parent / "adapter", is_trainable=True)
        print(f"Continuing from {parent}: epoch {total_epochs}, reusing verified parent evaluation", flush=True)
    training = train(model, tokenizer, training_batches=batches, updates=math.ceil(len(examples) / 4), learning_rate=1e-4)
    training["epochs"] = 1
    training["action_examples_seen"] = sum(len(batch) for batch in batches)
    dump(output / "training.json", training)
    model.save_pretrained(output / "adapter", safe_serialization=True)
    probe = encode_prompt(tokenizer, examples[0]["prompt"]).input_ids.to("cuda")
    expected = probe_logits(model, probe)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base(SNAPSHOT), output / "adapter", is_trainable=False)
    restored = probe_logits(reloaded, probe)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    after = evaluate(reloaded, tokenizer, cases, "after", output)
    report = {
        "model": MODEL_ID, "revision": REVISION, "device": torch.cuda.get_device_name(), "rank": 16,
        "epochs": 1, "total_epochs": total_epochs, "seed": seed,
        "parent_run": str(parent) if parent else None, "optimizer_state": "fresh AdamW",
        "before_evaluation": "verified parent after.json" if parent else "fresh base evaluation",
        "parent_adapter_sha256": parent_report["adapter_sha256"] if parent_report else None,
        "next_step_hints": False, "rule_fallback": False,
        "training": training, "before": before["summary"], "after": after["summary"],
        "reload_logits_max_abs_difference": (restored - expected).abs().max().item(),
        "train_sha256": sha256(DATASET / "train-actions.jsonl"), "eval_sha256": sha256(DATASET / "eval-episodes.jsonl"),
        "adapter_sha256": sha256(output / "adapter/adapter_model.safetensors"),
    }
    dump(output / "report.json", report)
    print(json.dumps({"before": report["before"], "after": report["after"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    run(args.output, args.parent)
