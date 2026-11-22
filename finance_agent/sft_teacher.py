"""Train a fresh LoRA on verified teacher actions and measure real CUDA throughput.

Default: 20 optimizer updates (4 single-example microbatches per update).
Mixed selection starts with the longest samples, then covers groups and random
examples without replacement. No API calls, evaluation, padding or truncation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
from random import Random
from threading import Event, Thread
from time import perf_counter

from .deepseek_trajectories import atomic_json, atomic_text, dumps, read_jsonl, run_lock
from .filings import ROOT
from .inspect_trajectories import SNAPSHOT
from .prepare_sft import sha256
from .training_checkpoint import read_checkpoint, restore_training_state, save_checkpoint

RELEASE = ROOT / "data/sft/deepseek-20261113"
OUTPUT = ROOT / "data/runtime/sft-teacher-20step"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def encode_example(tokenizer, row, max_length=8192):
    if row.get("split") != "train" or row.get("gold_visible") is not False:
        raise ValueError("Only verified blind training actions are accepted")
    if "expected" in row or "oracle_actions" in row:
        raise ValueError("Audit labels must not enter the SFT input")
    if len(row["completion"]) != 1 or row["completion"][0]["role"] != "assistant":
        raise ValueError("Expected one assistant completion")
    rendered = tokenizer.apply_chat_template(row["prompt"], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer(rendered, add_special_tokens=False).input_ids
    target = tokenizer(row["completion"][0]["content"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    if not prefix or not target or tokenizer.eos_token_id is None:
        raise ValueError("Empty prompt/target or missing EOS")
    if len(prefix) + len(target) > max_length:
        raise ValueError(f"{row['case_id']} turn {row['turn']}: context exceeds {max_length}; no truncation performed")
    return {"case_id": row["case_id"], "turn": row["turn"], "group": row["group"],
            "sequence": prefix + target, "target": target, "prefix_length": len(prefix)}


def select_examples(examples, steps, accumulation, seed=42, selection="mixed"):
    count = min(len(examples), steps * accumulation)
    if steps > math.ceil(len(examples) / accumulation):
        raise ValueError("Requested updates exceed one pass through the dataset")
    rng = Random(seed)
    indexes = list(range(len(examples)))
    if selection == "random":
        rng.shuffle(indexes)
        return [examples[i] for i in indexes[:count]]
    if selection != "mixed":
        raise ValueError("Unknown selection policy")
    # The first optimizer step is a reproducible long-context memory stress test.
    longest = sorted(indexes, key=lambda i: len(examples[i]["sequence"]), reverse=True)[:min(accumulation, count)]
    selected, used = list(longest), set(longest)
    candidates = [i for i in indexes if i not in used]
    rng.shuffle(candidates)
    groups = sorted({e["group"] for e in examples})
    for group in groups:
        if len(selected) >= count:
            break
        match = next((i for i in candidates if i not in used and examples[i]["group"] == group), None)
        if match is not None:
            selected.append(match)
            used.add(match)
    # Include the largest vocabulary projection target as well as longest context.
    largest_target = max(indexes, key=lambda i: len(examples[i]["target"]))
    if largest_target not in used and len(selected) < count:
        selected.append(largest_target)
        used.add(largest_target)
    selected.extend(i for i in candidates if i not in used)
    return [examples[i] for i in selected[:count]]


def batches(examples, accumulation):
    for index in range(0, len(examples), accumulation):
        yield examples[index:index + accumulation]


def target_window(example):
    """Target logits start at the last prefix token; prior messages have no loss."""
    length, target = len(example["sequence"]), len(example["target"])
    if example["prefix_length"] != length - target or target < 1 or length <= target:
        raise ValueError("Invalid causal target boundary")
    return length - target - 1, length - 1


def completion_loss(model, example, device, torch):
    target_window(example)
    ids = torch.tensor([example["sequence"]], dtype=torch.long, device=device)
    target = torch.tensor(example["target"], dtype=torch.long, device=device)
    # Keep just the final target positions and one extra causal position. Avoid
    # projecting all 7K context tokens into the 152K-token vocabulary.
    logits = model(input_ids=ids, logits_to_keep=len(example["target"]) + 1, use_cache=False).logits
    return torch.nn.functional.cross_entropy(logits[0, :-1].float(), target)


def segment_end(completed, total, segment_updates=None):
    if not 0 <= completed <= total or total <= 0:
        raise ValueError('Invalid completed/total update count')
    if segment_updates is not None and segment_updates <= 0:
        raise ValueError('segment_updates must be positive')
    return min(total, completed + segment_updates) if segment_updates else total


class DeviceMemory:
    def __init__(self, torch):
        self.torch, self.stop = torch, Event()
        self.peak_used = 0
        self.thread = Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            free, total = self.torch.cuda.mem_get_info()
            self.peak_used = max(self.peak_used, total - free)
            self.stop.wait(.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def run(release, output, snapshot, *, steps=20, accumulation=4, max_length=8192,
        learning_rate=1e-4, seed=42, warmup_steps=2, selection="mixed", checkpoint_every=50, resume=False,
        segment_updates=None, rank=16, alpha=32, dropout=0.0, target_modules=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_modules = list(TARGET_MODULES if target_modules is None else target_modules)
    if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1 or not target_modules or len(set(target_modules)) != len(target_modules):
        raise ValueError('Invalid LoRA settings')
    if steps <= 0 or accumulation <= 0 or checkpoint_every <= 0 or not 0 <= warmup_steps < steps or not 0 < learning_rate <= 1:
        raise ValueError('Invalid training schedule')
    segment_end(0, steps, segment_updates)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; benchmark artifacts are preserved")
    if resume and not (output / "latest-checkpoint.json").exists():
        raise ValueError("No saved checkpoint to resume")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 is required")
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    data_file = release / "train-actions.jsonl"
    if sha256(data_file) != manifest["files_sha256"]["train-actions.jsonl"]:
        raise ValueError("Teacher data differs from verified release")
    settings = {"model": "Qwen/Qwen3-1.7B", "snapshot": str(snapshot), "release": str(release),
                "train_actions_sha256": sha256(data_file), "trainer_sha256": sha256(Path(__file__)),
                "steps": steps, "microbatch_size": 1, "gradient_accumulation": accumulation,
                "max_length": max_length, "learning_rate": learning_rate, "seed": seed,
                "warmup_steps_excluded_from_speed": warmup_steps, "selection": selection,
                "rank": rank, "alpha": alpha, "dropout": dropout, "target_modules": target_modules,
                "base_dtype": "bfloat16", "attention": "sdpa", "gradient_checkpointing": True,
                "loss_scope": "completion and EOS only; all prompt tokens masked",
                "example_weighting": "equal weight per action; task_weight metadata not applied",
                "checkpoint_every": checkpoint_every,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}}
    started = perf_counter()
    with run_lock(output):
        checkpoint, prior = read_checkpoint(output, settings) if resume else (None, None)
        if resume and (output / "pause.requested").exists():
            raise ValueError("Remove pause.requested before resuming this run")
        if not resume:
            atomic_json(output / "config.json", settings)
            (output / "trainer-source.py").write_bytes(Path(__file__).read_bytes())
        source_snapshot = output / f"trainer-source-{settings['trainer_sha256'][:12]}.py"
        if not source_snapshot.exists():
            source_snapshot.write_bytes(Path(__file__).read_bytes())
        settings["trainer_source_snapshot"] = str(source_snapshot)
        atomic_json(output / "status.json", {"state": "preparing", "at": datetime.now(timezone.utc).isoformat()})
        try:
            torch.manual_seed(seed)
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            examples = [encode_example(tokenizer, row, max_length) for row in read_jsonl(data_file)]
            selected = select_examples(examples, steps, accumulation, seed, selection)
            training_batches = list(batches(selected, accumulation))
            if len(training_batches) != steps:
                raise ValueError("Requested optimizer update count does not match selected data")
            source_tokens = sum(len(e["sequence"]) for e in examples)
            source_targets = sum(len(e["target"]) for e in examples)
            plan = {"source_actions": len(examples), "source_sequence_tokens": source_tokens,
                    "source_supervised_tokens": source_targets, "selected_actions": len(selected),
                    "selected_sequence_tokens": sum(len(e["sequence"]) for e in selected),
                    "selected_groups": dict(Counter(e["group"] for e in selected)),
                    "source_max_length": max(len(e["sequence"]) for e in examples),
                    "selected_max_length": max(len(e["sequence"]) for e in selected),
                    "selected_max_completion": max(len(e["target"]) for e in selected),
                    "first_step_is_long_context_stress": selection == "mixed",
                    "examples": [{k: e[k] for k in ("case_id", "turn", "group", "prefix_length")} |
                                 {"length": len(e["sequence"]), "supervised_tokens": len(e["target"])} for e in selected]}
            if resume:
                old_plan = json.loads((output / "sampling.json").read_text(encoding="utf-8"))
                if old_plan != plan:
                    raise ValueError("Resume example order/tokenization changed")
            else:
                atomic_json(output / "sampling.json", plan)
            print(dumps({k: v for k, v in plan.items() if k != "examples"}), flush=True)
            free, total = torch.cuda.mem_get_info()
            baseline_used = total - free
            model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
            model.config.use_cache = False
            if resume:
                model = PeftModel.from_pretrained(model, checkpoint / "adapter", is_trainable=True)
            else:
                model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                    target_modules=target_modules, task_type="CAUSAL_LM"))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            parameters = [p for p in model.parameters() if p.requires_grad]
            if resume and prior["trainable_parameter_names"] != [name for name, p in model.named_parameters() if p.requires_grad]:
                raise ValueError("Trainable parameter ordering differs from the saved optimizer state")
            optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
            if resume:
                restore_training_state(checkpoint, optimizer, torch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            records = list(prior["records"]) if prior else []
            completed = prior["completed_updates"] if prior else 0
            stop_at = segment_end(completed, steps, segment_updates)
            loss = None
            if resume:
                old_log = output / "steps.jsonl"
                if old_log.exists():
                    logged = read_jsonl(old_log)
                    if [r["update"] for r in logged] != [r["update"] for r in records]:
                        archive = output / f"steps-before-resume-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
                        archive.write_bytes(old_log.read_bytes())
                        atomic_text(old_log, "".join(dumps(r) + "\n" for r in records))
            checkpoint_seconds = float(prior["progress"].get("checkpoint_seconds", 0)) if prior else 0.0
            historical_peak_allocated = float(prior["progress"].get("peak_allocated_mib", 0)) if prior else 0.0
            historical_peak_reserved = float(prior["progress"].get("peak_reserved_mib", 0)) if prior else 0.0
            def commit_checkpoint():
                nonlocal checkpoint_seconds
                torch.cuda.synchronize()
                checkpoint_start = perf_counter()
                path = save_checkpoint(output, model, optimizer, torch, settings, records, {
                    "checkpoint_seconds": checkpoint_seconds,
                    "peak_allocated_mib": max(historical_peak_allocated, torch.cuda.max_memory_allocated() / 2**20),
                    "peak_reserved_mib": max(historical_peak_reserved, torch.cuda.max_memory_reserved() / 2**20)})
                checkpoint_seconds += perf_counter() - checkpoint_start
                print(f"checkpoint committed: {path.name}", flush=True)
                return path
            preparation_seconds = perf_counter() - started
            training_start = perf_counter()
            with DeviceMemory(torch) as memory, (output / "steps.jsonl").open("a" if resume else "x", encoding="utf-8") as log:
                for update, batch in enumerate(training_batches, 1):
                    if update <= completed:
                        continue
                    torch.cuda.synchronize()
                    step_start = perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    value = 0.0
                    for example in batch:
                        loss = completion_loss(model, example, "cuda", torch)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite loss at update {update}")
                        (loss / len(batch)).backward()
                        value += loss.item() / len(batch)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize()
                    duration = perf_counter() - step_start
                    free, total = torch.cuda.mem_get_info()
                    record = {"update": update, "loss": value, "grad_norm": norm.item(),
                              "seconds": duration, "actions": len(batch),
                              "sequence_tokens": sum(len(e["sequence"]) for e in batch),
                              "supervised_tokens": sum(len(e["target"]) for e in batch),
                              "max_sequence_length": max(len(e["sequence"]) for e in batch),
                              "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                              "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
                              "cuda_reported_used_mib": round((total - free) / 2**20, 1),
                              "examples": [{"case_id": e["case_id"], "turn": e["turn"]} for e in batch]}
                    log.write(dumps(record) + "\n")
                    log.flush()
                    records.append(record)
                    atomic_json(output / "status.json", {"state": "training", "completed_updates": update,
                                                           "total_updates": steps, "last_step": record})
                    print(f"update {update}/{steps}: loss={value:.4f} time={duration:.2f}s "
                          f"max_length={record['max_sequence_length']} allocated={record['allocated_mib']:.1f}MiB", flush=True)
                    requested = (output / "pause.requested").exists()
                    pausing = requested or update >= stop_at
                    if update % checkpoint_every == 0 or update == steps or pausing:
                        committed = commit_checkpoint()
                    if pausing and update < steps:
                        atomic_json(output / "status.json", {"state": "paused", "completed_updates": update,
                            "total_updates": steps, "checkpoint": str(committed), "gpu_released_on_process_exit": True,
                            "pause_reason": "requested" if requested else "segment_limit"})
                        print(f"Paused safely at update {update}; resume from the committed checkpoint.", flush=True)
                        return {"state": "paused", "completed_updates": update}
            training_seconds = perf_counter() - training_start
            peak_allocated = max(historical_peak_allocated, torch.cuda.max_memory_allocated() / 2**20)
            peak_reserved = max(historical_peak_reserved, torch.cuda.max_memory_reserved() / 2**20)
            if not any(p.detach().count_nonzero().item() for name, p in model.named_parameters() if "lora_B" in name):
                raise RuntimeError("LoRA parameters did not update")
            model.save_pretrained(output / "adapter", safe_serialization=True)
            probe = torch.tensor([selected[0]["sequence"][:64]], device="cuda")
            with torch.inference_mode():
                expected = model.eval()(input_ids=probe, logits_to_keep=1, use_cache=False).logits.float().cpu()
            del optimizer, parameters, loss, model
            gc.collect()
            torch.cuda.empty_cache()
            base = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
            restored_model = PeftModel.from_pretrained(base, output / "adapter", is_trainable=False).eval()
            with torch.inference_mode():
                restored = restored_model(input_ids=probe, logits_to_keep=1, use_cache=False).logits.float().cpu()
            torch.testing.assert_close(restored, expected, rtol=0, atol=0)
            steady = records[warmup_steps:]
            steady_seconds = sum(r["seconds"] for r in steady)
            steady_tokens = sum(r["sequence_tokens"] for r in steady)
            throughput = steady_tokens / steady_seconds if steady_seconds else None
            report = {"config": settings, "completed_updates": len(records), "action_examples_seen": len(selected),
                      "preparation_seconds": preparation_seconds, "training_seconds": training_seconds,
                      "summed_update_seconds": sum(r["seconds"] for r in records), "checkpoint_seconds": checkpoint_seconds,
                      "resumed_from_update": completed, "latest_checkpoint": str(output / "latest-checkpoint.json"),
                      "steady_updates": len(steady), "steady_sequence_tokens_per_second": throughput,
                      "steady_seconds_per_update": steady_seconds / len(steady) if steady else None,
                      "peak_process_allocated_mib": peak_allocated, "peak_process_reserved_mib": peak_reserved,
                      "cuda_reported_used_before_model_mib": baseline_used / 2**20,
                      "sampled_peak_cuda_reported_used_mib": memory.peak_used / 2**20, "device_total_mib": total / 2**20,
                      "memory_note": "Allocator peaks belong to this PyTorch process. cuda.mem_get_info values are recorded separately; on Windows WDDM they need not match nvidia-smi total device use.",
                      "selected_max_length": plan["selected_max_length"], "source_max_length": plan["source_max_length"],
                      "estimated_epoch_training_seconds": source_tokens / throughput if throughput else None,
                      "estimate_note": "Token-throughput projection, not measured epoch time; excludes evaluation and checkpoint overhead. Length distribution and shared GPU load affect it.",
                      "reload_logits_max_abs_difference": (restored - expected).abs().max().item(),
                      "adapter_sha256": sha256(output / "adapter/adapter_model.safetensors"),
                      "total_wall_seconds": perf_counter() - started, "task_evaluation_performed": False,
                      "gpu": torch.cuda.get_device_name(), "steps": records}
            atomic_json(output / "report.json", report)
            atomic_json(output / "status.json", {"state": "complete", "completed_updates": len(records),
                                                   "report": str(output / "report.json")})
            print(json.dumps({k: v for k, v in report.items() if k not in ("config", "steps")}, ensure_ascii=False, indent=2), flush=True)
            return report
        except BaseException as error:
            atomic_json(output / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                                                   "elapsed_seconds": perf_counter() - started})
            atomic_json(output / "status.json", {"state": "failed", "error_type": type(error).__name__})
            raise


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--steps", type=positive, default=20)
    parser.add_argument("--accumulation", type=positive, default=4)
    parser.add_argument("--max-length", type=positive, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--selection", choices=("mixed", "random"), default="mixed")
    parser.add_argument("--checkpoint-every", type=positive, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--segment-updates", type=positive)
    parser.add_argument("--rank", type=positive, default=16)
    parser.add_argument("--alpha", type=positive, default=32)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--target-modules", nargs='+', default=TARGET_MODULES)
    args = parser.parse_args()
    if not 0 <= args.warmup_steps < args.steps or not 0 < args.learning_rate <= 1:
        parser.error("warmup-steps must be less than steps; learning-rate must be positive and <=1")
    run(args.release.resolve(), args.output.resolve(), args.snapshot.resolve(), steps=args.steps,
        accumulation=args.accumulation, max_length=args.max_length, learning_rate=args.learning_rate,
        seed=args.seed, warmup_steps=args.warmup_steps, selection=args.selection,
        checkpoint_every=args.checkpoint_every, resume=args.resume, segment_updates=args.segment_updates,
        rank=args.rank, alpha=args.alpha, dropout=args.dropout, target_modules=args.target_modules)
