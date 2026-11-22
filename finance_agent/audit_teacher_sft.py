"""Re-execute exported teacher trajectories before promoting a training release."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .deepseek_trajectories import atomic_json, atomic_writer, dumps, initial_messages, job_key, now, read_jsonl
from .prepare_sft import sha256
from .trajectory_runtime import Environment, score_episode
from .trajectory_sources import OUTPUT


def replay_teacher(case, trajectory, data, train_reports=None):
    if train_reports is None:
        train_reports = set(json.loads((data / "manifest.json").read_text(encoding="utf-8"))["splits"]["train"]["reports"])
    messages = trajectory["messages"]
    expected_messages = initial_messages(case)
    if messages[:2] != expected_messages:
        raise ValueError("Initial messages do not match the blind student protocol")
    env, actions, observations, steps = Environment(case, data), [], [], []
    selected_year = None
    index, final = 2, None
    while index < len(messages):
        message = messages[index]
        if set(message) != {"role", "content"} or message["role"] != "assistant":
            raise ValueError("Expected an assistant JSON action")
        action = json.loads(message["content"])
        if case["group"] == "conditional_bridge" and action.get("tool") == "search" and action.get("period") == "H1":
            if selected_year is None or action.get("year") != selected_year:
                raise ValueError("Dependent search preceded or ignored observed argmax")
        steps.append({"prompt": list(expected_messages), "completion": [message], "turn": len(actions) + 1,
                      "action_type": action.get("tool", "finish")})
        actions.append(action)
        observation = env.execute(action)
        expected_messages.append(message)
        index += 1
        if action.get("done"):
            final = action
            if case["group"] == "cutoff" and not any(
                    a.get("tool") == "search" and f"{a.get('company_code')}-{a.get('year')}-{a.get('period')}" in case["required_reports"]
                    for a in actions):
                raise ValueError("Unanswerable cutoff task never searched the requested report scope")
            if index != len(messages):
                raise ValueError("Messages continue after the final answer")
            break
        if index >= len(messages):
            raise ValueError("Missing tool observation")
        expected_observation = {"role": "user", "content": "工具观察：" + dumps(observation)}
        if observation.get("tool") == "search" and any(hit["report_id"] not in train_reports for hit in observation["hits"]):
            raise ValueError("A held-out report entered the teacher training context")
        if messages[index] != expected_observation:
            raise ValueError("Saved observation differs from actual tool execution")
        expected_messages.append(expected_observation)
        observations.append(observation)
        if observation.get("operation") == "argmax":
            selected_year = observation["selected_year"]
        index += 1
    checks = score_episode(case, final, actions, observations)
    if not checks["passed"] or final != trajectory["answer"]:
        raise ValueError("Teacher trajectory failed independent replay or answer comparison")
    return steps


def audit(run, data, output):
    cases = {case["id"]: case for case in read_jsonl(run / "task-pool/questions.jsonl")}
    source_manifest = json.loads((run / "task-pool/manifest.json").read_text(encoding="utf-8"))
    if sha256(data / "manifest.json") != source_manifest["source_manifest_sha256"]:
        raise ValueError("Original dataset manifest changed")
    if sha256(data / "corpus.sqlite3") != source_manifest["source_corpus_sha256"]:
        raise ValueError("Public filing index changed")
    for name, digest in source_manifest["heldout_sha256"].items():
        if sha256(data / name) != digest:
            raise ValueError("Original held-out split changed")
    accepted, rejected, action_count, teacher_counts, groups = [], [], 0, Counter(), Counter()
    train_reports = set(json.loads((data / "manifest.json").read_text(encoding="utf-8"))["splits"]["train"]["reports"])
    primary = read_jsonl(run / "sft/train-trajectories.jsonl")
    with atomic_writer(output / "train-actions.jsonl") as actions, atomic_writer(output / "train-trajectories.jsonl") as trajectories:
        for original in primary:
            case = cases[original["case_id"]]
            alternatives = [original]
            for phase in sorted(run.glob("[0-9][0-9]-*"), reverse=True):
                path = phase / "episodes" / (job_key(case["id"], 0) + ".json")
                if phase.name == original.get("source_phase") or not path.exists():
                    continue
                row = json.loads(path.read_text(encoding="utf-8"))
                if row["checks"]["passed"] and row["split"] == "train":
                    alternatives.append({"case_id": row["case_id"], "group": row["group"], "split": "train",
                                         "protocol": row["protocol"], "teacher_model": row["teacher"]["model"],
                                         "gold_visible": row["gold_visible"], "provenance": "teacher_api_rollout",
                                         "source_phase": phase.name, "messages": row["messages"], "answer": row["answer"]})
            chosen = None
            for candidate in alternatives:
                try:
                    if candidate["gold_visible"] is not False or candidate["provenance"] != "teacher_api_rollout":
                        raise ValueError("Teacher provenance or blind-generation flag is missing")
                    steps = replay_teacher(case, candidate, data, train_reports)
                    chosen = candidate
                    break
                except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                    rejected.append({"case_id": case["id"], "group": case["group"],
                                     "source_phase": candidate.get("source_phase"), "error": str(error)})
            if chosen is None:
                continue
            trajectory = chosen
            metadata = {key: value for key, value in trajectory.items() if key not in ("messages", "answer")}
            for step in steps:
                actions.write(dumps({**metadata, **step, "task_weight": 1 / len(steps)}) + "\n")
                action_count += 1
            trajectories.write(dumps(trajectory) + "\n")
            accepted.append(case["id"])
            groups[case["group"]] += 1
            teacher_counts[trajectory["teacher_model"]] += 1
            if len(accepted) % 100 == 0:
                print(f"Independent teacher replays passed: {len(accepted)}", flush=True)
    atomic_json(output / "quarantine.json", rejected)
    accepted_cases = [cases[case_id] for case_id in accepted]
    coverage = {"companies": sorted({company["code"] for case in accepted_cases for company in case["companies"]}),
                "required_reports": sorted({report for case in accepted_cases for report in case["required_reports"]}),
                "distinct_gold_fact_ids": len({ref for case in accepted_cases for ref in case["expected"]["citations"]}),
                "metric_families": ["revenue", "profit", "cash"],
                "answer_statuses": dict(Counter(case["expected"]["status"] for case in accepted_cases))}
    manifest = {"audited_at": now(), "source_run": str(run), "verified_task_pool": len(cases),
                "teacher_trajectories": len(accepted), "sft_actions": action_count, "quarantined_candidates": len(rejected),
                "excluded_tasks": len(primary) - len(accepted), "audit_code_sha256": sha256(Path(__file__)),
                "all_exported_trajectories_reexecuted": True, "conditional_dependencies_checked": True,
                "heldout_hashes_unchanged": True, "gold_visible": False, "reference_actions_in_release": 0,
                "by_group": dict(groups), "by_teacher": dict(teacher_counts),
                "source_coverage": coverage,
                "task_ids": accepted, "files_sha256": {name: sha256(output / name) for name in (
                    "train-actions.jsonl", "train-trajectories.jsonl", "quarantine.json")},
                "limitations": ["Program-derived financial questions on three metrics and nine task families.",
                                "This verifies source consistency, execution and grading; no full manual annotation audit was performed.",
                                "Public external QA seeds are separate and are not part of this teacher release."]}
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key not in ("task_ids", "files_sha256")}, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve(), args.data.resolve(), args.output.resolve())
