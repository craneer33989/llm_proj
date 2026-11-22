"""Prepare auditable RL-only prefix pools for the gated four-arm experiment."""
import argparse
import json
from pathlib import Path
import random

from .download import sha256
from .reopd import Rollout, digest, environment_lock, parse, render
from .reopd_compare import episode_table
from .reopd_model import MODELS, checkpoint
from .reopd_run import verify_lock, write_json, write_rows
from .reopd_train import approved_prefixes
from .search import DEFAULT_DATA, SearchEngine
from .training import TaskStore


def trajectory_prefixes(records, source, include_initial=False):
    prefixes = []
    for key, episode in sorted(records.items()):
        if episode["split"] != "rl" or key[1] != 0:
            raise ValueError(f"Expected one RL baseline trajectory per question: {key}")
        if episode["reason"] == "backend_error" or (source == "teacher" and not episode["supported_success"]):
            continue
        history = []
        for action in episode["steps"]:
            if source == "teacher" or history or include_initial:
                prefixes.append({"qid": episode["qid"], "split": "rl", "turn": len(history), "source": source,
                                 "prefix": list(history), "source_trajectory_id": digest({field: episode[field]
                                 for field in ("qid", "split", "attempt", "messages", "steps", "reason", "calls")}),
                                 "teacher_target": action["raw"] if source == "teacher" else None})
            if action["kind"] != "search" or "observation" not in action:
                break
            history.append(action)
    return prefixes


def build_pools(teacher, student, recovery, seed, ordinary_per_recovery):
    rng = random.Random(seed)
    if not recovery or not teacher:
        raise ValueError("Both vetted recovery cases and teacher prefixes are required")
    recovery = sorted(recovery, key=lambda row: row["qid"])
    keys = {(row["qid"], row["turn"]) for row in recovery}
    if any(parse(row["teacher_target"])[0] not in {"search", "finish"} for row in [*teacher, *recovery]):
        raise ValueError("A requires a valid teacher action on every training state")
    remaining = [row for row in student if (row["qid"], row["turn"]) not in keys]
    rng.shuffle(remaining)
    matched = []
    for target in sorted(recovery, key=lambda row: (row["turn"], row["prompt_tokens"], row["qid"])):
        candidates = [row for row in remaining if row["turn"] == target["turn"]]
        if not candidates:
            raise ValueError(f"Insufficient ordinary student prefixes at turn {target['turn']}")
        control = min(candidates, key=lambda row: abs(row["prompt_tokens"] - target["prompt_tokens"]))
        remaining.remove(control)
        matched.append({**control, "matched_recovery_qid": target["qid"],
                        "prompt_token_gap": abs(control["prompt_tokens"] - target["prompt_tokens"])})
    banned = {row["qid"] for row in [*recovery, *matched]}
    ordinary = [row for row in teacher if row["qid"] not in banned]
    count = ordinary_per_recovery * len(recovery)
    if len(ordinary) < count:
        raise ValueError(f"Only {len(ordinary)}/{count} distinct ordinary teacher states after matching")
    common = rng.sample(ordinary, count)
    directed = [*common, *recovery]
    rng.shuffle(directed)
    controls = {row["matched_recovery_qid"]: row for row in matched}
    mixed = [row if row["source"] == "teacher" else controls[row["qid"]] for row in directed]
    replay = [row for row in teacher if rng.random() < 0.6 ** row["turn"]]
    rng.shuffle(replay)
    return {"A": directed, "B": replay, "C": mixed, "D": directed}


def prepare(data, run, seed, ordinary_per_recovery, failure_type="early_stop"):
    from transformers import AutoTokenizer

    engine, store = SearchEngine(data), TaskStore(data)
    try:
        lock = environment_lock(engine)
        if json.loads((run / "environment.json").read_text(encoding="utf-8")) != lock:
            raise ValueError("Baseline and pool index differ")
        paths = {source: run / f"rl-{name}" for source, name in (("student", "sft"), ("teacher", "teacher"))}
        tables = {source: episode_table(path) for source, path in paths.items()}
        if tables["student"].keys() != tables["teacher"].keys():
            raise ValueError("Teacher and student baseline cohort differs")
        for source, name in (("student", "sft"), ("teacher", "teacher")):
            baseline = json.loads((paths[source] / "manifest.json").read_text(encoding="utf-8"))
            if baseline["model"] != list(MODELS[name]) or baseline["split"] != "rl":
                raise ValueError(f"Baseline manifest is not the pinned {name} RL run")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint("sft"), local_files_only=True)

        def measured(records):
            for row in records:
                env = Rollout(engine, store.tasks["rl"][row["qid"]], "rl", row["prefix"])
                yield {**row, "prompt_tokens": len(tokenizer.encode(render(env.messages), add_special_tokens=False))}

        suffix = "-wrong-without-search" if failure_type == "wrong_without_search" else ""
        recovery_dir = run / ("rl-recovery" + suffix)
        recovery = list(measured(approved_prefixes(recovery_dir)))
        teacher = list(measured(trajectory_prefixes(tables["teacher"], "teacher")))
        student = list(measured(trajectory_prefixes(tables["student"], "student", include_initial=bool(suffix))))
        pools = build_pools(teacher, student, recovery, seed, ordinary_per_recovery)
        out = run / f"ablation-pools-{seed}{suffix}"
        out.mkdir(exist_ok=True)
        inputs = [recovery_dir / filename for filename in ("audit.csv", "prefixes.jsonl")]
        inputs.extend(path / filename for path in paths.values() for filename in ("manifest.json", "episodes.jsonl", "scores.jsonl"))
        verify_lock(out / "manifest.json", {"environment": lock, "models": MODELS, "seed": seed,
                    "ordinary_per_recovery": ordinary_per_recovery, "retention_base": 0.6, "failure_type": failure_type,
                    "input_sha256": {path.relative_to(run).as_posix(): sha256(path) for path in inputs},
                    "pool_sha256": {group: digest(pool) for group, pool in pools.items()}})
        for group, pool in pools.items():
            write_rows(out / f"{group}.jsonl", pool)
        gaps = [row["prompt_token_gap"] for row in pools["C"] if row["source"] == "student"]
        write_json(out / "metrics.json", {"states": {group: len(pool) for group, pool in pools.items()},
                   "shared_ordinary_states": ordinary_per_recovery * len(recovery),
                   "matched_states": len(gaps), "mean_prompt_token_gap": sum(gaps) / len(gaps),
                   "max_prompt_token_gap": max(gaps)})
        return out
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ordinary-per-recovery", type=int, default=3)
    parser.add_argument("--failure-type", choices=["early_stop", "wrong_without_search"], default="early_stop")
    args = parser.parse_args()
    if args.ordinary_per_recovery < 1:
        parser.error("--ordinary-per-recovery must be positive")
    print(prepare(args.data, args.run, args.seed, args.ordinary_per_recovery, args.failure_type))


if __name__ == "__main__":
    main()
