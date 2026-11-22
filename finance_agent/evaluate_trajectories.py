"""Evaluate full tool rollouts; reference replay is explicitly separate from model scores."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from .expand_sft import read_jsonl
from .prepare_sft import sha256
from .trajectory_runtime import Environment, dumps, initial_messages, score_episode
from .trajectory_sources import OUTPUT


def model_generator(snapshot, adapter=None, max_new_tokens=384):
    # Lazy imports allow data validation without CUDA or model dependencies.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()

    @torch.inference_mode()
    def generate(messages):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        if inputs.input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
            raise ValueError("Context exceeds the model limit; evaluation does not silently truncate evidence")
        result = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generate


def rollout(case, policy, output=OUTPUT, generate=None):
    env, messages = Environment(case, output), initial_messages(case)
    actions, observations, invalid, final = [], [], [], None
    start = perf_counter()
    for turn in range(case["max_actions"]):
        raw = None
        try:
            if policy == "model":
                raw = generate(messages)
                action = json.loads(raw.strip())
            elif policy == "reference":
                action = case["oracle_actions"][turn]
            elif policy == "early-stop":
                action = case["oracle_actions"][0] if turn == 0 else {"done": True, "status": "insufficient", "citations": []}
            else:
                raise ValueError(f"Unknown policy: {policy}")
            if not isinstance(action, dict):
                raise ValueError("Model action must be a JSON object")
            actions.append(action)
            if not action.get("done") and action.get("tool") == "search" and env.searches >= case["max_searches"]:
                raise ValueError("Search budget exceeded")
            observation = env.execute(action)
            messages.append({"role": "assistant", "content": dumps(action)})
            if action.get("done"):
                final = action
                break
            observations.append(observation)
            messages.append({"role": "user", "content": "工具观察：" + dumps(observation)})
        except (ValueError, KeyError, TypeError, ArithmeticError) as error:
            invalid.append({"turn": turn + 1, "error": str(error), "raw": raw})
            break  # No oracle repair, rule fallback, or completion of missing model actions.
    checks = score_episode(case, final, actions, observations)
    checks["valid_actions"] = not invalid
    checks["passed"] = checks["passed"] and not invalid
    return {"case_id": case["id"], "group": case["group"], "checks": checks, "actions": actions,
            "observations": observations, "invalid": invalid, "elapsed_seconds": round(perf_counter() - start, 3)}


def summarize(rows):
    groups = sorted({r["group"] for r in rows})
    numeric = [r for r in rows if r["checks"]["numeric_correct"] is not None]
    return {"cases": len(rows), "passed": sum(r["checks"]["passed"] for r in rows),
        "by_group": {g: {"cases": sum(r["group"] == g for r in rows),
                         "passed": sum(r["group"] == g and r["checks"]["passed"] for r in rows)} for g in groups},
        "numeric": {"correct": sum(r["checks"]["numeric_correct"] for r in numeric), "total": len(numeric)},
        "citations_correct": sum(r["checks"]["citations_correct"] for r in rows),
        "invalid_episodes": sum(not r["checks"]["valid_actions"] for r in rows),
        "unfinished_episodes": sum(not r["checks"]["finished"] for r in rows),
        "premature_stops": sum(r["checks"]["premature_stop"] for r in rows),
        "extra_searches": sum(r["checks"]["extra_searches"] for r in rows),
        "average_searches": round(sum(r["checks"]["searches"] for r in rows) / len(rows), 3)}


def evaluate(split="test", policy="reference", output=OUTPUT, destination=None, generate=None, limit=None):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for filename in (f"{split}-questions.jsonl", "sources.json", "corpus.sqlite3"):
        if sha256(output / filename) != manifest["files_sha256"][filename]:
            raise ValueError(f"Evaluation input changed: {filename}")
    questions = read_jsonl(output / f"{split}-questions.jsonl")
    if limit:
        questions = questions[:limit]
    if not questions:
        raise ValueError("No evaluation questions")
    rows = []
    for i, case in enumerate(questions, 1):
        row = rollout(case, policy, output, generate)
        rows.append(row)
        if i % 10 == 0:
            print(f"{policy} {split}: {i}/{len(questions)}", flush=True)
    report = {"policy": policy, "split": split, "protocol": manifest["protocol"], "rule_fallback": False,
        "model_performance": policy == "model", "dataset_manifest_sha256": sha256(output / "manifest.json"),
        "summary": summarize(rows), "rows": rows}
    destination = destination or output / "evaluation" / f"{split}-{policy}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--policy", choices=["reference", "early-stop", "model"], default="reference")
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.policy == "model" and not args.snapshot:
        parser.error("--policy model requires --snapshot (a local model directory)")
    generate = model_generator(args.snapshot, args.adapter) if args.policy == "model" else None
    evaluate(args.split, args.policy, args.data.resolve(), args.output, generate, args.limit)
