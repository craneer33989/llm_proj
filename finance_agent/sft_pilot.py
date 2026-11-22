"""Run ten local LoRA updates and compare unassisted search before and after."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
from random import Random
from time import perf_counter

from peft import LoraConfig, PeftModel, get_peft_model
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .agent import MODEL_ID, REVISION, ask
from .corpus import DB
from .evaluate import score
from .expand_sft import EXPANDED, read_jsonl
from .filings import ROOT, RUNTIME
from .model_service import encode_prompt, generate_text
from .prepare_sft import OUTPUT, sha256

SEED = 42
STEPS = 10
LR = 3e-4
SNAPSHOT = ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots" / REVISION
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_base(snapshot):
    return AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, dtype=torch.bfloat16).to("cuda")


def evaluate(model, tokenizer, questions, stage):
    model.eval()

    def generate(messages):
        return generate_text(model, tokenizer, messages, "cuda")

    rows = []
    for i, case in enumerate(questions, 1):
        answer = ask(case["question"], case["company_code"], case["as_of"], mode="agent", generate=generate)
        checks = score(case, answer)
        searches = [step for step in answer["trace"] if step["action"] == "search"]
        checks["second_search"] = case["group"] == "bridge" and len(searches) >= 2
        checks["second_hop_hit"] = checks["second_search"] and any(
            f"{hit['report_id']}:{hit['page']}" == case["sources"][1] for hit in searches[1]["hits"]
        )
        rows.append({"case": case, "checks": checks, "answer": answer})
        print(f"{stage} {i}/{len(questions)} {case['id']} pass={checks['passed']} "
              f"second_search={checks['second_search']} second_hit={checks['second_hop_hit']}", flush=True)
    summary = {}
    for group in sorted({row["case"]["group"] for row in rows}):
        subset = [row for row in rows if row["case"]["group"] == group]
        summary[group] = {"cases": len(subset)} | {
            key: sum(bool(row["checks"][key]) for row in subset)
            for key in ("passed", "second_search", "second_hop_hit", "both_hops_retrieved", "stopped_after_first_hop")
        }
    return {"mode": "agent", "rule_fallback": False, "summary": summary, "rows": rows}


def batches(tokenizer):
    new = read_jsonl(EXPANDED / "train-actions.jsonl")
    audited = {row["case_id"]: row for row in read_jsonl(OUTPUT / "failure-replays.jsonl")}
    seeds = []
    for sample in read_jsonl(OUTPUT / "sft-seed.jsonl"):
        row = audited[sample["case_id"]]
        if row["split"] != "train":
            raise ValueError("A diagnostic sample was included in training")
        seeds.append(dict(sample, turn=len(row["original_trace"]), group=row["group"]))
    pools = {
        "direct": [row for row in new + seeds if row["group"] == "direct"],
        "bridge_start": [row for row in new if row["group"] == "bridge" and row["turn"] == 1],
        "continuation": [row for row in new + seeds if row["turn"] == 2],
    }
    rng = Random(SEED)
    for pool in pools.values():
        rng.shuffle(pool)
    positions = Counter()
    for _ in range(STEPS):
        batch = []
        for group in ("direct", "bridge_start", "continuation", "continuation"):
            pool = pools[group]
            row = pool[positions[group] % len(pool)]
            positions[group] += 1
            prefix = encode_prompt(tokenizer, row["prompt"]).input_ids[0].tolist()
            content = row["completion"][0]["content"]
            target = tokenizer(content, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            if not prefix or not target or len(prefix) + len(target) > 1024:
                raise ValueError(f"{row['case_id']}: empty or oversized example")
            batch.append((row["case_id"], row["turn"], prefix + target, target))
        yield batch


def train(model, tokenizer, *, training_batches=None, updates=STEPS, learning_rate=LR):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    losses, start = [], perf_counter()
    for step, batch in enumerate(batches(tokenizer) if training_batches is None else training_batches, 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _, _, sequence, expected in batch:
            ids = torch.tensor([sequence], device="cuda")
            targets = torch.tensor(expected, device="cuda")
            # Last context position predicts the first completion token.
            logits = model(input_ids=ids, logits_to_keep=len(expected) + 1, use_cache=False).logits
            loss = F.cross_entropy(logits[0, :-1].float(), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at update {step}")
            (loss / len(batch)).backward()
            step_loss += loss.item() / len(batch)
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append({"update": step, "loss": step_loss, "grad_norm": norm.item(),
                       "examples": [{"case_id": row[0], "turn": row[1]} for row in batch]})
        print(f"LoRA update {step}/{updates}: loss={step_loss:.4f}", flush=True)
    updated = any(
        parameter.count_nonzero().item() for name, parameter in model.named_parameters() if "lora_B" in name
    )
    if len(losses) != updates or not updated:
        raise RuntimeError("LoRA weights did not update")
    return {
        "updates": len(losses), "microbatches_per_update": 4,
        "losses": losses, "learning_rate": learning_rate,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "gpu_free_at_training_start_mib": round(free / 2**20, 1),
        "gpu_total_mib": round(total / 2**20, 1),
        "training_seconds": round(perf_counter() - start, 2),
    }


@torch.inference_mode()
def probe_logits(model, ids):
    return model.eval()(input_ids=ids, logits_to_keep=1, use_cache=False).logits.float().cpu()


def run(output, snapshot):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required for this pilot")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    diagnostic_file = EXPANDED / "diagnostic-questions.jsonl"
    questions = read_jsonl(diagnostic_file)
    data_manifest = json.loads((EXPANDED / "manifest.json").read_text(encoding="utf-8"))
    if data_manifest["corpus_sha256"] != sha256(DB):
        raise ValueError("The current index differs from the verified diagnostic index")
    if not questions or {row["company_code"] for row in questions} != {data_manifest["diagnostic_company"]}:
        raise ValueError("Expected a nonempty held-out issuer diagnostic")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    base = load_base(snapshot)
    before = evaluate(base, tokenizer, questions, "before")
    dump(output / "before.json", before)
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0, target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
        revision=REVISION,
    ))
    training = train(model, tokenizer)
    dump(output / "training.json", training)
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    probe = read_jsonl(EXPANDED / "train-actions.jsonl")[0]["prompt"]
    ids = encode_prompt(tokenizer, probe).input_ids.to("cuda")
    reference = probe_logits(model, ids)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base(snapshot), adapter, is_trainable=False)
    restored = probe_logits(reloaded, ids)
    torch.testing.assert_close(restored, reference, rtol=0, atol=0)
    after = evaluate(reloaded, tokenizer, questions, "after_reload")
    dump(output / "after.json", after)
    report = {
        "model": MODEL_ID, "revision": REVISION, "device": torch.cuda.get_device_name(),
        "seed": SEED, "rank": 16, "target_modules": TARGET_MODULES, "base_dtype": "bfloat16",
        "diagnostic_sha256": sha256(diagnostic_file), "train_actions_sha256": sha256(EXPANDED / "train-actions.jsonl"),
        "sft_seed_sha256": sha256(OUTPUT / "sft-seed.jsonl"),
        "adapter_file": str(adapter / "adapter_model.safetensors"),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "reload_logits_max_abs_difference": (restored - reference).abs().max().item(),
        "training": training, "before": before["summary"], "after": after["summary"],
        "rule_fallback": False,
    }
    dump(output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("before", "after", "reload_logits_max_abs_difference")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUNTIME / "sft-pilot")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    run(args.output, args.snapshot)
