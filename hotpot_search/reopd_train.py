"""Local LoRA smoke and resource-gated four-arm trainer (not slime reproduction)."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

from .download import sha256
from .reopd import Rollout, digest, environment_lock, parse, render
from .reopd_model import Actor, MODELS
from .reopd_run import verify_lock, write_json, write_rows
from .search import DEFAULT_DATA, SearchEngine, rows
from .training import TaskStore


def action_logprobs(model, prompt, action):
    import torch

    if prompt.ndim != 1 or action.ndim != 1 or not len(prompt) or not len(action):
        raise ValueError("Expected nonempty, unpadded prompt and action token vectors")
    inputs = torch.cat((prompt, action)).unsqueeze(0)
    positions = torch.arange(len(prompt) - 1, len(inputs[0]) - 1, device=inputs.device)
    logits = model(input_ids=inputs, use_cache=False, logits_to_keep=positions).logits[0].float()
    return logits.log_softmax(dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)


def sampled_reverse_kl(student_lp, teacher_lp):
    if student_lp.ndim != 1 or student_lp.shape != teacher_lp.shape:
        raise ValueError("Token-aligned one-dimensional teacher and student scores required")
    return ((student_lp - teacher_lp).detach() * student_lp).mean()


def parameter_hash(model, trainable_only):
    import torch

    result = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not trainable_only or parameter.requires_grad:
                result.update(name.encode())
                result.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy())
    return result.hexdigest()


def approved_prefixes(pool_dir):
    failure_type = "wrong_without_search" if pool_dir.name == "rl-recovery-wrong-without-search" else "early_stop"
    turn = 0 if failure_type == "wrong_without_search" else 1
    with (pool_dir / "audit.csv").open(encoding="utf-8", newline="") as stream:
        approved = {(entry["qid"], entry["source_trajectory_id"], entry.get("recovery_evidence_id"))
                    for entry in csv.DictReader(stream)
                    if entry["human_review"] == "approved" and (entry.get("review_note") or "").strip()}
    selected = [record for record in rows(pool_dir / "prefixes.jsonl")
                if (record["qid"], record["source_trajectory_id"], record.get("recovery_evidence_id")) in approved]
    qids = {record["qid"] for record in selected}
    if len(selected) != len(qids):
        raise ValueError("The recovery pool contains duplicate training question IDs")
    if len(qids) < 20:
        raise ValueError(f"P2 gate not met: {len(qids)}/20 independently reviewed training questions")
    if any(record["split"] != "rl" or not record["eligible"] or record["teacher_successes"] < 2
           or record.get("teacher_scorable") != 3 or not record.get("recovery_evidence_id")
           or record["failure_type"] != failure_type or record["turn"] != turn
           or len(record["prefix"]) != turn or parse(record["teacher_target"])[0] != "search" for record in selected):
        raise ValueError("Invalid approved recovery prefix")
    return selected


def verify_tokenizer(student, teacher):
    if (student.get_vocab() != teacher.get_vocab() or student.all_special_ids != teacher.all_special_ids
            or student.eos_token_id != teacher.eos_token_id or student.pad_token_id != teacher.pad_token_id
            or student.backend_tokenizer.to_str() != teacher.backend_tokenizer.to_str()):
        raise ValueError("Teacher and student tokenizer/serialization mismatch")


def experiment_schedule(run, group, seed, env_lock, failure_type="early_stop"):
    suffix = "-wrong-without-search" if failure_type == "wrong_without_search" else ""
    dev = run / ("smoke-dev" + suffix)
    for filename in ("manifest.json", "paired_metrics.json", "fixed/paired_metrics.json"):
        if not (dev / filename).is_file():
            raise ValueError("P4 requires completed P3 paired dev and fixed-prefix evaluation")
    dev_manifest = json.loads((dev / "manifest.json").read_text(encoding="utf-8"))
    if dev_manifest["training_recipe"].get("failure_type", "early_stop") != failure_type:
        raise ValueError("P3 failure type does not match P4")
    paired = json.loads((dev / "paired_metrics.json").read_text(encoding="utf-8"))
    comparison = paired.get("comparisons", {}).get("D_minus_A", {}).get("supported_success", {})
    if len(dev_manifest.get("specs", [])) < 10 or comparison.get("paired_prefixes", 0) < 10:
        raise ValueError("P4 requires at least ten scored paired P3 dev episodes")
    for arm in ("A", "D"):
        probe = run / f"smoke-{arm}-1step-{dev_manifest['training_recipe']['seed']}{suffix}" / "metrics.json"
        metrics = json.loads(probe.read_text(encoding="utf-8"))
        if metrics["steps"] != 1 or not metrics["adapter_reloaded"] or not metrics["adapter_changed"]:
            raise ValueError(f"Missing successful P3 one-step probe for {arm}")
        if arm == "D" and not metrics["teacher_unchanged"]:
            raise ValueError("One-step D probe did not verify teacher immutability")
    directory = run / f"ablation-pools-{seed}{suffix}"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["seed"] != seed or manifest["environment"] != env_lock or manifest.get("failure_type", "early_stop") != failure_type:
        raise ValueError("Ablation pool seed or environment mismatch")
    for name, expected_hash in manifest["input_sha256"].items():
        if sha256(run / name) != expected_hash:
            raise ValueError(f"Stale ablation pool input: {name}")
    schedule = list(rows(directory / f"{group}.jsonl"))
    if not schedule or digest(schedule) != manifest["pool_sha256"][group]:
        raise ValueError(f"Missing or modified group-{group} prefix schedule")
    return schedule, sha256(manifest_path)


def update_action(action, used, budget):
    return action if budget is None else action[:budget - used]


def train(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict

    pool = args.run / ("rl-recovery" if args.failure_type == "early_stop" else "rl-recovery-wrong-without-search")
    selected = approved_prefixes(pool)
    engine, store = SearchEngine(args.data), TaskStore(args.data)
    env_lock = environment_lock(engine)
    verify_lock(args.run / "environment.json", env_lock)
    if args.update_tokens is None:
        schedule = sorted(selected, key=lambda record: record["qid"])
        random.Random(args.seed).shuffle(schedule)
        output = args.run / f"smoke-{args.group}-{args.steps}step-{args.seed}"
        if args.failure_type != "early_stop":
            output = output.with_name(output.name + "-wrong-without-search")
        pool_manifest_sha = None
    else:
        schedule, pool_manifest_sha = experiment_schedule(args.run, args.group, args.seed, env_lock, args.failure_type)
        output = args.run / f"ablation-{args.group}-{args.update_tokens}tokens-{args.seed}"
        if args.failure_type == "wrong_without_search":
            output = output.with_name(output.name + "-wrong-without-search")
    if any(record["split"] != "rl" or record["qid"] not in store.tasks["rl"] for record in schedule):
        raise ValueError("Only RL questions are allowed in the recovery trainer")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The local 4B probe requires BF16 CUDA")
    output.mkdir(exist_ok=False)
    verify_lock(output / "manifest.json", {"environment": env_lock, "group": args.group, "student": MODELS["sft"],
                                          "teacher": MODELS["teacher"], "seed": args.seed, "steps": args.steps,
                                          "failure_type": args.failure_type,
                                          "max_tokens": args.max_tokens, "lr": args.lr, "rank": 8, "alpha": 16,
                                          "update_token_budget": args.update_tokens,
                                          "updates_qids": [schedule[index % len(schedule)]["qid"] for index in range(args.steps)]
                                          if args.update_tokens is None else [row["qid"] for row in schedule],
                                          "ablation_pool_manifest_sha256": pool_manifest_sha,
                                          "pool_sha256": sha256(pool / "prefixes.jsonl"),
                                          "audit_sha256": sha256(pool / "audit.csv")})
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    student = Actor("sft", args.max_tokens)
    student.model = get_peft_model(student.model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
                                   target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    student.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    initial = parameter_hash(student.model, trainable_only=True)
    teacher = Actor("teacher", args.max_tokens, device="cpu") if args.group != "A" else None
    if teacher:
        verify_tokenizer(student.tokenizer, teacher.tokenizer)
        teacher.model.requires_grad_(False)
        teacher_initial = parameter_hash(teacher.model, trainable_only=False)
    optimizer = torch.optim.AdamW([param for param in student.model.parameters() if param.requires_grad], lr=args.lr)
    updates = []
    used = 0
    while (len(updates) < args.steps if args.update_tokens is None else used < args.update_tokens):
        index = len(updates)
        row = schedule[index % len(schedule)]
        env = Rollout(engine, store.tasks["rl"][row["qid"]], "rl", row["prefix"])
        optimizer.zero_grad(set_to_none=True)
        if args.group == "A":
            prompt = torch.tensor(student.tokenizer.encode(render(env.messages), add_special_tokens=False), device=device)
            action = torch.tensor(student.tokenizer.encode(row["teacher_target"], add_special_tokens=False), device=device)
            if len(action) > args.max_tokens:
                raise ValueError("SFT teacher action exceeds the action budget")
            teacher_lp, teacher_tokens = None, 0
            raw = row["teacher_target"]
        else:
            student.model.eval()
            encoded, sampled = student.sample([env.messages], temperature=1.0)
            prompt, action = encoded.input_ids[0].clone(), sampled[0].clone()
            raw = student.tokenizer.decode(action, skip_special_tokens=True)
        action_length = len(action)
        action = update_action(action, used, args.update_tokens)
        if teacher is not None:
            with torch.no_grad():
                teacher_lp = action_logprobs(teacher.model, prompt.cpu(), action.cpu()).to(device)
            teacher_tokens = len(prompt) + len(action)
        student.model.train()
        student_lp = action_logprobs(student.model, prompt, action)
        loss = -student_lp.mean() if teacher_lp is None else sampled_reverse_kl(student_lp, teacher_lp)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite student loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(student.model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        used += len(action)
        updates.append({"step": index + 1, "qid": row["qid"], "loss": loss.item(), "grad_norm": float(grad),
                        "source": row["source"], "turn": row["turn"], "action_tokens_before_budget_clip": action_length,
                        "student_update_tokens": len(action), "prompt_masked_tokens": len(prompt),
                        "teacher_scoring_input_tokens": teacher_tokens, "sampled_reverse_kl":
                        None if teacher_lp is None else (student_lp - teacher_lp).detach().mean().item(),
                        "sampled_action_kind": parse(raw)[0], "policy_search_calls": 0,
                        "prefix_replay_calls": len(env.env.session.trace)})
        if index == 0:
            write_json(output / "mask-example.json", {"prefix_tokens": prompt.tolist(), "action_tokens": action.tolist(),
                                                       "loss_mask": [0] * len(prompt) + [1] * len(action)})
        write_rows(output / "updates.jsonl", updates)
        print(updates[-1], flush=True)

    final = parameter_hash(student.model, trainable_only=True)
    if final == initial:
        raise RuntimeError("LoRA weights did not change")
    if teacher and parameter_hash(teacher.model, trainable_only=False) != teacher_initial:
        raise RuntimeError("Frozen teacher weights changed")
    checkpoint_dir = output / "adapter"
    student.model.save_pretrained(checkpoint_dir)
    expected = {name: weight.detach().cpu().clone() for name, weight in get_peft_model_state_dict(student.model).items()}
    del student, optimizer, teacher, loss, student_lp, grad
    torch.cuda.empty_cache()
    loaded = Actor("sft", args.max_tokens, adapter=checkpoint_dir)
    restored = get_peft_model_state_dict(loaded.model)
    if restored.keys() != expected.keys() or any(not torch.equal(value, restored[name].cpu()) for name, value in expected.items()):
        raise RuntimeError("Reloaded adapter differs from saved weights")
    write_json(output / "metrics.json", {"steps": len(updates), "student_update_tokens": used,
                                          "student_sampled_tokens": sum(u["action_tokens_before_budget_clip"] for u in updates)
                                          if args.group != "A" else 0,
                                          "prefix_replay_calls": sum(u["prefix_replay_calls"] for u in updates),
                                          "policy_search_calls": sum(u["policy_search_calls"] for u in updates),
                                          "teacher_scoring_input_tokens": sum(u["teacher_scoring_input_tokens"] for u in updates),
                                          "seconds": time.monotonic() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "adapter_reloaded": True,
                                          "adapter_sha256": {name: sha256(checkpoint_dir / name) for name in
                                                             ("adapter_config.json", "adapter_model.safetensors")},
                                          "adapter_changed": True, "teacher_unchanged": True if args.group != "A" else None})
    engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=["A", "B", "C", "D"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--steps", type=int, default=1)
    length.add_argument("--update-tokens", type=int)
    parser.add_argument("--confirm-p4-budget", action="store_true")
    parser.add_argument("--failure-type", choices=["early_stop", "wrong_without_search"], default="early_stop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if not 0 < args.lr < 1 or args.max_tokens < 1:
        parser.error("Require positive learning rate and action budget")
    if args.update_tokens is None and (args.group not in {"A", "D"} or not 1 <= args.steps <= 30):
        parser.error("Local smoke supports only A and D and 1..30 steps")
    if args.update_tokens is not None and (args.update_tokens < 1 or not args.confirm_p4_budget):
        parser.error("P4 requires a positive update token budget and explicit --confirm-p4-budget")
    train(args)


if __name__ == "__main__":
    main()
