"""Bounded unattended collection: audited task pool, teacher pilot, bulk and rescue."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import Random
from threading import Event, Lock, Thread
import json
import os
import re

from .build_trajectories import candidates, replay
from .deepseek_trajectories import (
    APIError, DeepSeekClient, Settings, atomic_json, atomic_text, atomic_writer,
    dumps, export_results, generate_episode, job_key, now, prepare, read_jsonl, run_lock,
)
from .prepare_sft import sha256
from .trajectory_sources import OUTPUT


def load_api_key(path=None):
    if path is None:
        value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    else:
        text = path.read_text(encoding="utf-8-sig").strip()
        matches = set(re.findall(r"\bsk-[A-Za-z0-9_-]{16,}\b", text))
        if len(matches) != 1:
            raise ValueError("The specified key file must contain exactly one sk- API key; its contents were not logged.")
        value = matches.pop()
    return value


def balanced(cases):
    pools = defaultdict(list)
    for case in cases:
        pools[case["group"]].append(case)
    rng = Random(20261113)
    for pool in pools.values():
        rng.shuffle(pool)
    result = []
    while any(pools.values()):
        for name in sorted(pools):
            if pools[name]:
                result.append(pools[name].pop())
    return result


def expand_pool(data, output):
    """Keep the original train/validation/test files untouched; verify all candidates."""
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    # Validate the pinned corpus before constructing any new task.
    prepare(data, data / "train-questions.jsonl", "train", None, None, 1, Settings())
    protected = [f"{split}-{kind}.jsonl" for split in ("validation", "test")
                 for kind in ("questions", "actions", "trajectories")]
    protected_hashes = {name: sha256(data / name) for name in protected}
    train_codes = set(original["splits"]["train"]["companies"])
    seen, accepted, rejected = set(), [], []
    with atomic_writer(output / "reference-actions.jsonl") as actions, atomic_writer(output / "reference-trajectories.jsonl") as trajectories:
        for case in candidates(read_jsonl(data / "fact-catalog.jsonl")):
            if case["split"] != "train":
                continue
            if case["semantic_key"] in seen:
                raise ValueError("Duplicate semantic task in expanded pool")
            seen.add(case["semantic_key"])
            if not {company["code"] for company in case["companies"]} <= train_codes:
                raise ValueError("Expanded task contains a held-out issuer")
            try:
                records, audit = replay(case, data)
            except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                rejected.append({"case_id": case["id"], "group": case["group"], "error": str(error)})
                continue
            accepted.append(case)
            for record in records:
                actions.write(dumps({**record, "provenance": "programmatic_reference"}) + "\n")
            trajectories.write(dumps({**audit, "provenance": "programmatic_reference"}) + "\n")
            if len(accepted) % 100 == 0:
                print(f"Reference replay verified: {len(accepted)}", flush=True)
    cases = balanced(accepted)
    atomic_text(output / "questions.jsonl", "".join(dumps(case) + "\n" for case in cases))
    atomic_text(output / "rejected-candidates.jsonl", "".join(dumps(row) + "\n" for row in rejected))
    if protected_hashes != {name: sha256(data / name) for name in protected}:
        raise ValueError("Held-out data changed during expansion")
    manifest = {"created_at": now(), "source_manifest_sha256": sha256(data / "manifest.json"),
                "source_corpus_sha256": sha256(data / "corpus.sqlite3"),
                "candidate_tasks": len(seen), "verified_tasks": len(cases), "rejected_tasks": len(rejected),
                "reference_actions": sum(len(case["oracle_actions"]) for case in cases),
                "by_group": dict(Counter(case["group"] for case in cases)),
                "heldout_sha256": protected_hashes, "gold_visible": False,
                "reference_is_teacher_data": False,
                "limitations": ["Template families and three financial metrics are shared with the original task pool.",
                                "Reference replay verifies data/tools, not teacher or student performance."]}
    atomic_json(output / "manifest.json", manifest)
    return cases


class CollectionStopped(APIError):
    def __init__(self, reason):
        super().__init__(reason, fatal=True)


class Budget:
    """Reserve a pessimistic cost before every request, including retries.

    Estimates use peak, cache-miss USD/M token rates. Unknown/unreported requests
    retain their reservation; these are an execution guard, not a billing quote.
    """
    RATES = {"deepseek-v4-pro": (1.32, 3.96), "deepseek-flash": (0.30, 1.20)}

    def __init__(self, maximum_usd, deadline, output, timeout=90):
        self.maximum = maximum_usd
        self.deadline = deadline
        self.timeout = timeout
        self.output = output
        self.lock = Lock()
        self.reservations = {}
        self.reported_cost = 0.0
        self.requests = 0
        self.reason = None
        self.consecutive_errors = 0

    def snapshot(self):
        return {"updated_at": now(), "maximum_estimated_usd": self.maximum,
                "reported_peak_cache_miss_estimate_usd": round(self.reported_cost, 6),
                "reserved_or_unreported_usd": round(sum(self.reservations.values()), 6),
                "requests": self.requests, "deadline_utc": self.deadline.isoformat(), "stop_reason": self.reason,
                "rates_usd_per_million": self.RATES,
                "note": "Conservative estimate, not provider billing. Reservations cover requests without usable usage."}

    def observe(self, event, model):
        with self.lock:
            kind, request_id = event["event"], event.get("request_id")
            if kind == "request":
                if (self.output.parent / "stop.requested").exists():
                    self.reason = self.reason or "operator_requested_stop"
                if self.reason:
                    raise CollectionStopped(self.reason)
                if datetime.now(timezone.utc) + timedelta(seconds=self.timeout + 10) >= self.deadline:
                    self.reason = "deadline"
                input_rate, output_rate = self.RATES[model]
                # UTF-8 bytes plus chat overhead overestimates ordinary text tokens.
                input_bound = len(dumps(event["payload"]["messages"]).encode("utf-8")) + 2048
                reservation = (input_bound * input_rate + event["payload"]["max_tokens"] * output_rate) / 1_000_000
                if self.reported_cost + sum(self.reservations.values()) + reservation > self.maximum:
                    self.reason = self.reason or "estimated_cost_limit"
                if self.reason:
                    atomic_json(self.output, self.snapshot())
                    raise CollectionStopped(self.reason)
                self.reservations[request_id] = reservation
                self.requests += 1
            elif kind == "response":
                self.consecutive_errors = 0
                try:
                    usage = json.loads(event["body"])["usage"]
                    prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
                    if type(prompt) is not int or type(completion) is not int or min(prompt, completion) < 0:
                        raise ValueError("Invalid usage")
                    input_rate, output_rate = self.RATES[model]
                    self.reported_cost += (prompt * input_rate + completion * output_rate) / 1_000_000
                    self.reservations.pop(request_id, None)
                except (ValueError, KeyError, TypeError):
                    pass
            elif kind == "request_error":
                self.consecutive_errors += 1
                if event.get("http_status") in (400, 401, 402, 403, 404, 422):
                    self.reason = f"provider_http_{event['http_status']}"
                elif self.consecutive_errors >= 8:
                    self.reason = "repeated_network_or_provider_errors"
            else:
                return
            atomic_json(self.output, self.snapshot())


class BoundedClient(DeepSeekClient):
    def __init__(self, api_key, settings, budget):
        super().__init__(api_key, settings)
        self.budget = budget

    def complete(self, messages, record):
        def bounded_record(event):
            self.budget.observe(event, self.settings.model)
            record(event)
        return super().complete(messages, bounded_record)


def collect_phase(name, cases, data, output, api_key, budget, model, workers):
    phase = output / name
    phase.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "status.json", {"state": "running", "phase": name, "planned_tasks": len(cases),
                                           "budget": budget.snapshot(), "updated_at": now()})
    questions = phase / "questions.jsonl"
    atomic_text(questions, "".join(dumps(case) + "\n" for case in cases))
    settings = Settings(model=model, max_tokens=8192, timeout=90, retries=2)
    _, config = prepare(data, questions, "train", None, None, 1, settings)
    config["supervisor_sha256"] = sha256(Path(__file__))
    atomic_json(phase / "manifest.json", {"created_at": now(), "config": config})
    client = BoundedClient(api_key, settings, budget)
    pending = iter(cases)
    completed = 0
    # Submit only a bounded number of episodes. A stopped budget prevents new calls
    # in already-running episodes too, preserving all partial attempts.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for _ in range(workers):
            case = next(pending, None)
            if case is not None:
                futures[pool.submit(generate_episode, case, 0, data, phase, client, settings)] = case
        while futures:
            future = next(as_completed(futures))
            case = futures.pop(future)
            row = future.result()
            completed += 1
            print(f"{name}: {completed}/{len(cases)} {case['id']} {row['outcome']}", flush=True)
            if row["fatal_api_error"] and not budget.reason:
                budget.reason = row["api_error"]
            if completed % 10 == 0:
                export_results(phase, cases, 1)
            if not budget.reason:
                case = next(pending, None)
                if case is not None:
                    futures[pool.submit(generate_episode, case, 0, data, phase, client, settings)] = case
    summary = export_results(phase, cases, 1)
    atomic_json(output / "status.json", {"state": "running", "phase": name, "summary": summary, "budget": budget.snapshot()})
    return phase


def merge_phases(output, phases, total_tasks):
    """Choose one passing teacher trajectory per task; never mix reference actions."""
    selected, attempts = {}, []
    for phase in phases:
        for path in sorted((phase / "episodes").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            attempts.append({"case_id": row["case_id"], "group": row["group"], "outcome": row["outcome"],
                             "phase": phase.name, "checks": row["checks"]})
            if row["checks"]["passed"]:
                previous = selected.get(row["case_id"])
                if previous is None or len(row["actions"]) < len(previous[0]["actions"]):
                    selected[row["case_id"]] = (row, phase)
    target = output / "sft"
    actions_count = 0
    with atomic_writer(target / "train-actions.jsonl") as actions, atomic_writer(target / "train-trajectories.jsonl") as trajectories:
        for case_id, (row, phase) in sorted(selected.items()):
            metadata = {"case_id": case_id, "group": row["group"], "split": "train", "protocol": row["protocol"],
                        "teacher_model": row["teacher"]["model"], "gold_visible": False,
                        "provenance": "teacher_api_rollout", "source_phase": phase.name}
            trajectories.write(dumps({**metadata, "messages": row["messages"], "answer": row["answer"]}) + "\n")
            for step in row["steps"]:
                actions.write(dumps({**metadata, "turn": step["turn"], "action_type": step["action"].get("tool", "finish"),
                                     "prompt": step["prompt"], "completion": [{"role": "assistant", "content": dumps(step["action"])}]}) + "\n")
                actions_count += 1
    rejection_checks = Counter()
    failure_if_false = ("status_correct", "citations_correct", "numeric_correct", "calculation_correct",
                        "finished", "valid_actions", "budget_ok", "efficient_stop")
    for row in attempts:
        if row["outcome"] != "rejected":
            continue
        rejection_checks.update(name for name in failure_if_false if row["checks"].get(name) is False)
        if row["checks"].get("premature_stop"):
            rejection_checks["premature_stop"] += 1
    summary = {"updated_at": now(), "verified_task_pool": total_tasks, "saved_attempts": len(attempts),
               "attempted_tasks": len({r['case_id'] for r in attempts}), "accepted_teacher_trajectories": len(selected),
               "sft_actions": actions_count, "by_group": dict(Counter(row[0]["group"] for row in selected.values())),
               "by_teacher": dict(Counter(row[0]["teacher"]["model"] for row in selected.values())),
               "rejection_checks": dict(rejection_checks),
               "reference_actions_in_sft": 0}
    atomic_json(output / "summary.json", summary)
    atomic_text(output / "attempt-index.jsonl", "".join(dumps(row) + "\n" for row in attempts))
    return summary, set(selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minutes", type=int, default=105)
    parser.add_argument("--max-usd", type=float, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--api-key-file", type=Path, help="Read a user-specified key file in memory; never include its contents in logs")
    args = parser.parse_args(argv)
    if not 1 <= args.minutes <= 120 or not 1 <= args.workers <= 8 or not 0 < args.max_usd <= 100:
        parser.error("minutes must be 1..120, workers 1..8 and max-usd >0..100")
    data, output = args.data.resolve(), args.output.resolve()
    if any(output.glob("[0-9][0-9]-*/manifest.json")):
        parser.error("This output already contains teacher attempts; use a new directory to preserve its data and budget history")
    started = datetime.now(timezone.utc)
    deadline = started + timedelta(minutes=args.minutes)
    phases, total_tasks, budget = [], 0, None
    finished = Event()
    with run_lock(output):
        atomic_json(output / "status.json", {"state": "preparing", "started_at": started.isoformat(), "deadline_utc": deadline.isoformat()})
        try:
            cases = expand_pool(data, output / "task-pool")
            total_tasks = len(cases)
            if args.prepare_only:
                atomic_json(output / "status.json", {"state": "prepared", "verified_tasks": total_tasks,
                                                       "teacher_api_called": False, "prepared_at": now()})
                return 0
            api_key = load_api_key(args.api_key_file)
            if not api_key:
                atomic_json(output / "status.json", {"state": "waiting_for_api_key", "verified_tasks": total_tasks,
                                                       "teacher_api_called": False, "updated_at": now()})
                print("Verified task pool ready; run the launcher in the PowerShell session with DEEPSEEK_API_KEY.", flush=True)
                return 2
            budget = Budget(args.max_usd, deadline, output / "budget.json")
            atomic_json(output / "budget.json", budget.snapshot())

            def wall_clock_guard():
                # urllib's timeout is an idle socket timeout; a provider sending
                # keep-alive bytes can outlive it. Preserve completed files and
                # terminate the process if the absolute deadline is exceeded.
                seconds = max(0, (deadline - datetime.now(timezone.utc)).total_seconds())
                if finished.wait(seconds):
                    return
                budget.reason = "deadline"
                if finished.wait(20):
                    return
                try:
                    saved_phases = sorted(path for path in output.iterdir()
                                          if path.is_dir() and re.match(r"\d\d-", path.name)
                                          and (path / "questions.jsonl").exists())
                    for phase in saved_phases:
                        export_results(phase, read_jsonl(phase / "questions.jsonl"), 1)
                    summary, _ = merge_phases(output, saved_phases, total_tasks)
                    atomic_json(output / "status.json", {"state": "finished", "stop_reason": "wall_clock_deadline",
                                                           "summary": summary, "budget": budget.snapshot(), "finished_at": now()})
                finally:
                    os._exit(124)

            Thread(target=wall_clock_guard, daemon=True, name="collection-deadline").start()
            pilot = cases[:min(45, len(cases))]
            phases.append(collect_phase("01-pro-pilot", pilot, data, output, api_key, budget, "deepseek-v4-pro", args.workers))
            summary, solved = merge_phases(output, phases, total_tasks)
            if not budget.reason:
                pilot_rows = [json.loads(p.read_text(encoding="utf-8")) for p in (phases[0] / "episodes").glob("*.json")]
                valid_answers = sum(r["checks"]["numeric_correct"] is not False and r["checks"]["status_correct"] for r in pilot_rows)
                if not pilot_rows or valid_answers / len(pilot_rows) < 0.25:
                    budget.reason = "pilot_quality_gate_failed"
            remaining = [case for case in cases if case["id"] not in {c["id"] for c in pilot}]
            if remaining and not budget.reason:
                phases.append(collect_phase("02-flash-bulk", remaining, data, output, api_key, budget, "deepseek-flash", args.workers))
                summary, solved = merge_phases(output, phases, total_tasks)
            failed_ids = {row["case_id"] for phase in phases for row in read_jsonl(phase / "answers.jsonl")
                          if row["outcome"] == "rejected"} - solved
            rescue = balanced([case for case in cases if case["id"] in failed_ids])
            if rescue and not budget.reason:
                phases.append(collect_phase("03-pro-rescue", rescue, data, output, api_key, budget, "deepseek-v4-pro", args.workers))
            summary, _ = merge_phases(output, phases, total_tasks)
            atomic_json(output / "status.json", {"state": "finished", "stop_reason": budget.reason or "task_pool_finished",
                                                   "summary": summary, "budget": budget.snapshot(), "finished_at": now()})
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            finished.set()
            return 0
        except Exception as error:
            finished.set()
            if phases:
                merge_phases(output, phases, total_tasks)
            atomic_json(output / "status.json", {"state": "failed", "error_type": type(error).__name__,
                                                   "error": str(error), "updated_at": now(),
                                                   "budget": budget.snapshot() if budget else None})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
