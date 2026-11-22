"""Run and resume baseline/teacher recovery collection without opening test data."""
import argparse
import csv
import json
from pathlib import Path
import random
import time

from .reopd import BackendError, Rollout, digest, environment_lock
from .reopd_model import Actor, MODELS
from .search import DEFAULT_DATA, SearchEngine, rows
from .training import TaskStore


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def verify_lock(path, expected):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(expected)):
            raise ValueError(f"Run configuration changed: {path}")
    else:
        write_json(path, expected)


def reference_smoke(engine, store, out):
    task = store.tasks["rl"]["5ae23c595542996483e64937"]
    env = Rollout(engine, task, "rl")
    for query in ("America's Civil War magazine 1987", "Heeb Jewish website magazine"):
        env.advance("<tool_call>" + json.dumps({"name": "search", "arguments": {"query": query}}) + "</tool_call>")
    replay = Rollout(engine, task, "rl", env.steps)
    replay.advance(r"\boxed{America's Civil War}")
    episode, score = replay.record(store.gold["rl"][task["id"]], 0)
    reader = Rollout(engine, task, "rl", env.steps)
    document = reader.steps[-1]["doc_ids"][0]
    result = reader.env.step("read_document", {"document_id": document})
    observation = result["observation"]
    if observation.get("id") != document or not score["supported_success"]:
        raise RuntimeError("Reference failed: cannot read observed document or finish with complete evidence")
    write_json(out / "reference-smoke.json", {"scripted": True, "episode": episode, "scoring": score,
               "separate_read_check": {"doc_id": document, "observation_hash": digest(observation)}})


def run_batch(engine, store, actor, split, specs, temperature):
    envs = []
    for qid, attempt, prefix in specs:
        try:
            envs.append(Rollout(engine, store.tasks[split][qid], split, prefix))
        except BackendError:
            envs.append(None)
    while live := [index for index, env in enumerate(envs) if env is not None and not env.reason]:
        outputs = actor.generate([envs[index].messages for index in live], temperature)
        if len(outputs) != len(live):
            raise RuntimeError("Actor result count mismatch")
        for index, raw in zip(live, outputs, strict=True):
            try:
                envs[index].advance(raw)
            except BackendError:
                envs[index] = None
    return [env.record(store.gold[split][qid], attempt) if env else
            ({"qid": qid, "split": split, "attempt": attempt, "reason": "backend_error", "steps": [], "calls": 0}, None)
            for env, (qid, attempt, _) in zip(envs, specs, strict=True)]


def validate_batch(batch, specs, split):
    if ([(ep["qid"], ep["attempt"]) for ep, _ in batch] != [(qid, attempt) for qid, attempt, _ in specs]
            or any(ep["split"] != split or (score is None) != (ep["reason"] == "backend_error")
                   or (score is not None and (score["qid"], score["attempt"]) != (ep["qid"], ep["attempt"]))
                   for ep, score in batch)):
        raise ValueError("Batch episode/score manifest mismatch")


def resume_batch(batch, specs, split, seed, generate):
    validate_batch(batch, specs, split)
    for index, (episode, score) in enumerate(batch):
        if score is not None:
            continue
        retry_seed = int(digest(["backend-retry", seed, split, episode["qid"], episode["attempt"]])[:16], 16)
        retried = generate([specs[index]], retry_seed)
        validate_batch(retried, [specs[index]], split)
        if retried[0][1] is not None:
            batch[index] = retried[0]
    return batch


def reuse_baseline_batches(source, destination, target, lock):
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    source_lock = json.loads((source.parent / "environment.json").read_text(encoding="utf-8"))
    target_specs = json.loads(json.dumps(target["specs"]))
    original_specs = original["specs"]
    shared = {key: value for key, value in target.items() if key not in {"specs", "limit"}}
    if (source_lock != lock or any(original.get(key) != value for key, value in shared.items())
            or len(original_specs) >= len(target_specs) or original_specs != target_specs[:len(original_specs)]):
        raise ValueError("Original baseline is not a strict cohort prefix with identical inference settings")
    batch_size = target["batch"]
    reused = 0
    for offset in range(0, len(original_specs) - batch_size + 1, batch_size):
        name = f"batch-{offset:05}.json"
        old, new = source / name, destination / name
        if not old.is_file():
            continue
        batch = json.loads(old.read_text(encoding="utf-8"))
        validate_batch(batch, target_specs[offset:offset + batch_size], target["split"])
        if not new.exists():
            write_json(new, batch)
            reused += len(batch)
    return reused


def collect(engine, store, actor, split, specs, out, batch_size, temperature, seed):
    import torch

    out.mkdir(parents=True, exist_ok=True)
    episodes, scoring = [], []

    def generate(part, batch_seed):
        torch.manual_seed(batch_seed)
        return run_batch(engine, store, actor, split, part, temperature)

    for offset in range(0, len(specs), batch_size):
        part = specs[offset:offset + batch_size]
        path = out / f"batch-{offset:05}.json"
        if path.exists():
            batch = json.loads(path.read_text(encoding="utf-8"))
            if any(ep["reason"] == "backend_error" for ep, _ in batch):
                batch = resume_batch(batch, part, split, seed, generate)
                write_json(path, batch)
        else:
            batch = generate(part, seed + offset)
            validate_batch(batch, part, split)
            write_json(path, batch)
            print(f"{out.name}: {min(offset + batch_size, len(specs))}/{len(specs)}", flush=True)
        validate_batch(batch, part, split)
        episodes.extend(ep for ep, _ in batch)
        scoring.extend(score for _, score in batch if score is not None)
    write_rows(out / "episodes.jsonl", episodes)
    write_rows(out / "scores.jsonl", scoring)
    metrics = {"requested": len(specs), "scorable": len(scoring), "backend_errors": len(episodes) - len(scoring),
               "finished": sum(ep["reason"] == "finished" for ep in episodes),
               "calls": sum(ep["calls"] for ep in episodes)}
    for key in ("answer_em", "answer_f1", "support_doc_coverage", "support_sentence_recall",
                "answer_with_all_support_docs", "supported_success", "early_stop"):
        metrics[key + "_sum"] = sum(record[key] for record in scoring)
    write_json(out / "metrics.json", metrics)
    return episodes, scoring


def recovery_states(student, student_scores, failure_type):
    scores = {row["qid"]: row for row in student_scores}
    for episode in student:
        score = scores.get(episode["qid"])
        if score is None:
            continue
        if failure_type == "early_stop" and score["early_stop"]:
            yield episode, episode["steps"][:1]
        elif (failure_type == "wrong_without_search" and not score["answer_em"]
              and [step["kind"] for step in episode["steps"]] == ["finish"]):
            yield episode, []


def recovery_rows(student, student_scores, teacher, teacher_scores, failure_type="early_stop"):
    teacher_by_key = {(score["qid"], score["attempt"]): score for score in teacher_scores}
    teacher_episodes = {(episode["qid"], episode["attempt"]): episode for episode in teacher}
    if len(teacher_by_key) != len(teacher_scores) or len(teacher_episodes) != len(teacher):
        raise ValueError("Duplicate teacher recovery attempt")
    for ep, prefix in recovery_states(student, student_scores, failure_type):
        qid = ep["qid"]
        turn = len(prefix)
        ready, audit = [], []
        for attempt in range(3):
            key = qid, attempt
            rollout, score = teacher_episodes.get(key), teacher_by_key.get(key)
            if score is None:
                audit.append({"attempt": attempt, "status": rollout["reason"] if rollout else "missing"})
                continue
            if rollout is None or rollout["steps"][:turn] != prefix or len(rollout["steps"]) <= turn:
                raise ValueError(f"Teacher attempt is not aligned to the fixed student state: {key}")
            repaired = bool(score["supported_success"] and rollout["steps"][turn]["kind"] == "search")
            audit.append({"attempt": attempt, "status": "success" if repaired else "failed",
                          "trajectory_id": digest(rollout), "score_sha256": digest(score)})
            if repaired:
                ready.append((rollout, score))
        scorable = sum(record["status"] in {"success", "failed"} for record in audit)
        yield {"qid": qid, "split": "rl", "turn": turn, "source": "student", "attempt": ep["attempt"],
               "failure_type": failure_type, "prefix": prefix, "teacher_successes": len(ready),
               "source_trajectory_id": digest(ep), "recovery_evidence_id": digest(audit), "teacher_trial_audit": audit,
               "teacher_attempts": 3, "teacher_scorable": scorable, "eligible": scorable == 3 and len(ready) >= 2,
               "verified_doc_ids": ready[0][1]["visible_support_docs"] if ready else [],
               "teacher_target": ready[0][0]["steps"][turn]["raw"] if ready else None}


def recovery_review_cases(store, candidates, student, student_scores, teacher, teacher_scores):
    originals = {row["qid"]: row for row in student}
    originals_scored = {row["qid"]: row for row in student_scores}
    continuations = {(row["qid"], row["attempt"]): row for row in teacher}
    scored = {(row["qid"], row["attempt"]): row for row in teacher_scores}
    for candidate in candidates:
        qid = candidate["qid"]
        yield {"qid": qid, "source_trajectory_id": candidate["source_trajectory_id"],
               "recovery_evidence_id": candidate["recovery_evidence_id"], "eligible": candidate["eligible"],
               "gold": store.gold["rl"][qid], "student_episode": originals[qid], "student_score": originals_scored[qid],
               "teacher_trials": [{**trial, "episode": continuations.get((qid, trial["attempt"])),
                                   "score": scored.get((qid, trial["attempt"]))}
                                  for trial in candidate["teacher_trial_audit"]]}


def export_recovery(out, candidates):
    write_rows(out / "candidates.jsonl", candidates)
    accepted = [record for record in candidates if record["eligible"]]
    write_rows(out / "prefixes.jsonl", accepted)
    audit = out / "audit.csv"
    reviews = {}
    if audit.exists():
        with audit.open(encoding="utf-8", newline="") as stream:
            reviews = {(record["qid"], record["source_trajectory_id"], record.get("recovery_evidence_id")):
                       {"human_review": record["human_review"], "review_note": record.get("review_note", "")}
                       for record in csv.DictReader(stream)}
    review_keys = ("qid", "source_trajectory_id", "recovery_evidence_id")
    reviewed = [{**record, **reviews.get(tuple(record[key] for key in review_keys),
                                       {"human_review": "pending", "review_note": ""})}
                for record in candidates]
    with audit.with_suffix(".csv.tmp").open("w", encoding="utf-8", newline="") as stream:
        columns = [*review_keys, "turn", "failure_type", "teacher_successes", "teacher_attempts", "teacher_scorable",
                   "eligible", "human_review", "review_note"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reviewed)
    audit.with_suffix(".csv.tmp").replace(audit)
    resolved = sum(record["teacher_scorable"] == 3 for record in candidates)
    write_json(out / "pool-metrics.json", {"candidates": len(candidates), "scorable_candidates": resolved,
                                          "unresolved_candidates": len(candidates) - resolved,
                                          "verified_automatic": len(accepted),
                                          "recovery_pass_rate_scorable": len(accepted) / resolved if resolved else None,
                                          "reviewed_approved": sum(record["eligible"] and record["human_review"] == "approved"
                                                                   and bool(record["review_note"].strip())
                                                                   for record in reviewed),
                                          "minimum_distinct_qids": 20})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["reference", "baseline", "recover"])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["rl", "dev"], default="rl")
    parser.add_argument("--model", choices=list(MODELS), default="sft")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reuse-baseline-run", type=Path)
    parser.add_argument("--failure-type", choices=["early_stop", "wrong_without_search"], default="early_stop")
    args = parser.parse_args()
    if min(args.batch, args.limit, args.max_tokens) < 1:
        parser.error("--batch, --limit and --max-tokens must be positive")
    if args.stage == "recover" and args.split != "rl":
        parser.error("Recovery collection only uses the rl split")
    if args.reuse_baseline_run and (args.stage != "baseline" or args.split != "rl" or args.model == "diagnostic"
                                    or args.reuse_baseline_run.resolve() == args.run.resolve()):
        parser.error("Baseline reuse requires a separate source run and a formal RL baseline")
    args.run.mkdir(parents=True, exist_ok=True)
    engine, store = SearchEngine(args.data), TaskStore(args.data)
    lock = environment_lock(engine)
    verify_lock(args.run / "environment.json", lock)
    if args.stage == "reference":
        reference_smoke(engine, store, args.run)
        engine.close()
        return
    started = time.monotonic()
    qids = sorted(store.tasks[args.split])
    random.Random(args.seed).shuffle(qids)
    qids = qids[:args.limit]
    if args.stage == "baseline":
        actor_name, temperature = args.model, 0.0
        if actor_name == "diagnostic" and args.limit > 10:
            parser.error("The cached 1.7B checkpoint is a diagnostic only; limit is 10")
        out = args.run / f"{args.split}-{actor_name}"
        specs = [(qid, 0, []) for qid in qids]
    else:
        actor_name, temperature = "teacher", 0.7
        out = args.run / ("rl-recovery" if args.failure_type == "early_stop" else "rl-recovery-wrong-without-search")
        student_path = args.run / "rl-sft"
        student = list(rows(student_path / "episodes.jsonl"))
        scores = list(rows(student_path / "scores.jsonl"))
        if [ep["qid"] for ep in student] != qids:
            raise ValueError("Student baseline must cover the same seeded RL sample")
        specs = [(ep["qid"], attempt, prefix) for ep, prefix in recovery_states(student, scores, args.failure_type)
                 for attempt in range(3)]
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"model": list(MODELS[actor_name]), "seed": args.seed, "batch": args.batch,
                "limit": args.limit, "max_tokens": args.max_tokens, "split": args.split,
                "temperature": temperature, "specs": specs}
    if args.stage == "recover" and args.failure_type != "early_stop":
        manifest["failure_type"] = args.failure_type
    verify_lock(out / "manifest.json", manifest)
    if args.reuse_baseline_run:
        reused = reuse_baseline_batches(args.reuse_baseline_run / out.name, out, manifest, lock)
        print(f"{out.name}: reused {reused} episodes from identical complete batches", flush=True)
    actor = Actor(actor_name, args.max_tokens) if specs else None
    episodes, scoring = collect(engine, store, actor, args.split, specs, out, args.batch, temperature, args.seed)
    if args.stage == "recover":
        candidates = list(recovery_rows(student, scores, episodes, scoring, args.failure_type))
        export_recovery(out, candidates)
        write_rows(out / "recovery-cases.private.jsonl", recovery_review_cases(store, candidates, student, scores, episodes, scoring))
    print(f"{out}: {len(scoring)}/{len(specs)} scorable in {time.monotonic() - started:.1f}s", flush=True)
    engine.close()


if __name__ == "__main__":
    main()
