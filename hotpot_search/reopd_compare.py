"""Paired baseline export and post-reload development evaluation for the local smoke."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random

from .reopd import digest, environment_lock
from .download import sha256
from .reopd_model import Actor, MODELS
from .reopd_run import collect, recovery_states, verify_lock, write_json, write_rows
from .search import DEFAULT_DATA, SearchEngine, rows
from .training import TaskStore


def episode_table(path):
    def keyed(filename):
        table = {}
        for row in rows(path / filename):
            key = row["qid"], row["attempt"]
            if key in table:
                raise ValueError(f"Duplicate record in {path / filename}: {key}")
            table[key] = row
        return table

    episodes = keyed("episodes.jsonl")
    scores = keyed("scores.jsonl")
    if scores.keys() != {key for key, episode in episodes.items() if episode["reason"] != "backend_error"}:
        raise ValueError(f"Episode and score coverage disagree: {path}")
    return {key: {**episode, **scores.get(key, {})} for key, episode in episodes.items()}


def review_baseline(paths, out, store, split, seed, limit=30):
    groups = {}
    for model, path in paths.items():
        failures = sorted((key, record) for key, record in episode_table(path).items()
                          if record.get("supported_success") is False)
        random.Random(f"{seed}:{model}").shuffle(failures)
        groups[model] = iter(failures)
    cases = []
    while len(cases) < limit:
        previous = len(cases)
        for model, group in groups.items():
            if len(cases) >= limit or (sample := next(group, None)) is None:
                continue
            (qid, attempt), record = sample
            if record["split"] != split:
                raise ValueError(f"Case belongs to a different split: {qid}")
            gold = store.gold[split][qid]
            cases.append({"case_id": digest({"model": model, "record": record, "gold": gold}),
                          "model": model, "qid": qid, "attempt": attempt, "episode_and_score": record,
                          "gold": gold})
        if len(cases) == previous:
            break
    review_path = out / "failure-audit.csv"
    reviews = {}
    if review_path.exists():
        with review_path.open(encoding="utf-8", newline="") as stream:
            reviews = {row["case_id"]: row for row in csv.DictReader(stream)}
    fields = ["case_id", "model", "qid", "attempt", "failure_type", "review_note"]
    staged = review_path.with_suffix(".tmp")
    with staged.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            review = reviews.get(case["case_id"], {})
            writer.writerow({**case, "failure_type": review.get("failure_type", "pending"),
                             "review_note": review.get("review_note", "")})
    staged.replace(review_path)
    write_rows(out / "failure-cases.private.jsonl", cases)
    write_json(out / "failure-audit-metrics.json", {"seed": seed, "requested": limit, "sampled": len(cases),
                   "reviewed": sum(bool(reviews.get(case["case_id"], {}).get("review_note", "").strip())
                       and reviews[case["case_id"]]["failure_type"] in {"early_stop", "wrong_entity", "misread",
                                                                       "corpus_missing", "format_error", "other"}
                       for case in cases)})


def paired_interval(pairs, metric, seed=42, replicates=2000):
    clusters = defaultdict(list)
    for before, after in pairs:
        if before["qid"] != after["qid"] or before["attempt"] != after["attempt"]:
            raise ValueError("Unpaired evaluation records")
        if metric in before and metric in after:
            clusters[before["qid"]].append(float(after[metric]) - float(before[metric]))
    if not clusters:
        return {"paired_prefixes": 0, "distinct_questions": 0, "delta": None, "interval_95": None}
    groups = [(sum(group), len(group)) for group in clusters.values()]
    total = sum(count for _, count in groups)
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        sample = rng.choices(groups, k=len(groups))
        samples.append(sum(delta for delta, _ in sample) / sum(count for _, count in sample))
    samples.sort()
    return {"paired_prefixes": total, "distinct_questions": len(groups),
            "delta": sum(delta for delta, _ in groups) / total,
            "interval_95": [samples[int((replicates - 1) * quantile)] for quantile in (0.025, .975)]}


def compare(paths, out):
    tables = {name: episode_table(path) for name, path in paths.items()}
    reference = next(iter(tables))
    keys = list(tables[reference])
    if any(set(table) != set(keys) for table in tables.values()):
        raise ValueError("Comparison cohorts do not contain identical episode keys")
    columns = ["reason", "calls", "answer_em", "support_doc_coverage", "supported_success", "early_stop", "after_first_search"]
    with (out / "paired_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["qid", "attempt"] + [f"{name}_{key}" for name in tables for key in columns])
        writer.writeheader()
        for qid, attempt in keys:
            row = {"qid": qid, "attempt": attempt}
            for name, table in tables.items():
                record = table[qid, attempt]
                steps = record["steps"]
                after = steps[1]["kind"] if len(steps) > 1 and steps[0]["kind"] == "search" else None
                row.update({f"{name}_{key}": record.get(key) for key in columns})
                row[f"{name}_after_first_search"] = after
            writer.writerow(row)
    summary = {f"{after}_minus_{before}": {
        metric: paired_interval([(tables[before][key], tables[after][key]) for key in keys], metric)
        for metric in ("answer_em", "supported_success")}
        for index, before in enumerate(tables) for after in list(tables)[index + 1:]}
    write_json(out / "paired_metrics.json", {"reference": reference, "requested_pairs": len(keys), "comparisons": summary})


def ordinary_controls(run, out):
    positives = list(rows(run / "rl-recovery" / "prefixes.jsonl"))
    recovery_qids = {item["qid"] for item in rows(run / "rl-recovery" / "candidates.jsonl")}
    teacher = episode_table(run / "rl-teacher")
    controls = {record["qid"]: record for record in teacher.values()
                if record.get("supported_success") and len(record["steps"]) >= 2
                and record["steps"][0]["kind"] == "search" and record["qid"] not in recovery_qids}
    matched = []
    for positive in sorted(positives, key=lambda record: record["qid"]):
        if not controls:
            break
        length = len(positive["prefix"][0]["observation"])
        qid = min(controls, key=lambda key: (abs(len(controls[key]["steps"][0]["observation"]) - length), key))
        episode = controls.pop(qid)
        matched.append({"qid": qid, "split": "rl", "turn": 1, "source": "teacher", "matched_qid": positive["qid"],
                        "prefix": episode["steps"][:1], "teacher_target": episode["steps"][1]["raw"],
                        "source_trajectory_id": digest(episode)})
    write_rows(out / "ordinary-controls.jsonl", matched)
    write_json(out / "control-metrics.json", {"matched": len(matched), "target": len(positives),
                                              "match": "turn and nearest observation character count"})


def validate_adapters(adapters, env_lock, ablation):
    recipe = None
    checked = {}
    shared = ("seed", "max_tokens", "lr", "rank", "alpha", "pool_sha256", "audit_sha256")
    shared += ("update_token_budget", "ablation_pool_manifest_sha256") if ablation else ("steps", "updates_qids")
    for label, path in adapters.items():
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
        weights = {name: sha256(path / name) for name in ("adapter_config.json", "adapter_model.safetensors")}
        if (manifest["group"] != label or manifest["student"] != list(MODELS["sft"])
                or manifest["environment"] != env_lock or not metrics["adapter_reloaded"]
                or not metrics["adapter_changed"] or metrics["adapter_sha256"] != weights
                or (label != "A" and not metrics["teacher_unchanged"])):
            raise ValueError(f"Adapter is not a verified group-{label} checkpoint")
        if ablation and (not manifest["update_token_budget"] or not manifest["ablation_pool_manifest_sha256"]
                         or metrics["student_update_tokens"] != manifest["update_token_budget"]):
            raise ValueError(f"Group {label} did not complete the frozen student token budget")
        if not ablation and manifest.get("update_token_budget") is not None:
            raise ValueError("The smoke evaluator requires step-based A/D checkpoints")
        current = {key: manifest[key] for key in shared}
        current["failure_type"] = manifest.get("failure_type", "early_stop")
        if recipe is not None and recipe != current:
            raise ValueError("Checkpoint recipes disagree on the shared experiment settings")
        recipe = current
        checked[label] = weights
    return recipe, checked


def cohort_summary(paths, out, fixed_prefixes=False, failure_type="early_stop"):
    tables = {name: episode_table(path) for name, path in paths.items()}
    keys = list(tables["base"])
    natural = [tables["base"][key] for key in keys if "supported_success" in tables["base"][key]]
    easy = {key for key in keys if tables["base"][key].get("supported_success") and tables["base"][key]["calls"] <= 1}
    metrics, cases = {}, []
    for name, table in tables.items():
        scorable = [record for record in table.values() if "supported_success" in record]
        paired_easy = [table[key] for key in easy if "supported_success" in table[key]]
        metrics[name] = {"requested": len(keys), "distinct_questions": len({qid for qid, _ in keys}),
                         "scorable": len(scorable), "unresolved": len(keys) - len(scorable),
                         "supported_successes": sum(record["supported_success"] for record in scorable),
                         "answer_em_correct": sum(record["answer_em"] for record in scorable),
                         "mean_support_doc_coverage": sum(record["support_doc_coverage"] for record in scorable) / len(scorable)
                         if scorable else None, "mean_search_calls": sum(record["calls"] for record in scorable) / len(scorable)
                         if scorable else None, "easy_reference_cases": len(easy), "easy_paired": len(paired_easy),
                         "easy_regressions": sum(not record["supported_success"] for record in paired_easy)}
        if name == "base":
            continue
        for qid, attempt in keys:
            before, after = tables["base"][qid, attempt], table[qid, attempt]
            if "supported_success" in before and "supported_success" in after and before["supported_success"] != after["supported_success"]:
                cases.append({"arm": name, "qid": qid, "attempt": attempt,
                              "change": "improved" if after["supported_success"] else "regressed",
                              "before": before, "after": after})
    summary = {"arms": metrics, "cohort": "fixed_prefixes" if fixed_prefixes else "end_to_end"}
    if not fixed_prefixes:
        summary.update(natural_early_stop_count=sum(record["early_stop"] for record in natural),
                       natural_scorable_count=len(natural), failure_type=failure_type,
                       natural_target_failure_count=sum(1 for _ in recovery_states(natural, natural, failure_type)))
    write_json(out / "cohort_metrics.json", summary)
    write_rows(out / "changed_cases.jsonl", cases)


def copy_initial_cohort(source, destination, specs):
    if any(prefix for _, _, prefix in specs):
        raise ValueError("Only an empty-history cohort can reuse end-to-end episodes")
    selected = {(qid, attempt) for qid, attempt, _ in specs}
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("episodes.jsonl", "scores.jsonl"):
        write_rows(destination / filename, (row for row in rows(source / filename)
                                           if (row["qid"], row["attempt"]) in selected))


def evaluate_dev(args):

    engine, store = SearchEngine(args.data), TaskStore(args.data)
    env_lock = environment_lock(engine)
    verify_lock(args.run / "environment.json", env_lock)
    ablation = args.stage == "ablation-dev"
    adapters = {"A": args.adapter_a, "D": args.adapter_d} if not ablation else {
        "A": args.adapter_a, "B": args.adapter_b, "C": args.adapter_c, "D": args.adapter_d}
    recipe, checked = validate_adapters(adapters, env_lock, ablation)
    failure_type = recipe["failure_type"]
    out = args.run / (f"ablation-dev-{recipe['update_token_budget']}tokens-{recipe['seed']}" if ablation else "smoke-dev")
    if failure_type == "wrong_without_search":
        out = out.with_name(out.name + "-wrong-without-search")
    out.mkdir(exist_ok=True)
    qids = sorted(store.tasks["dev"])
    random.Random(args.seed).shuffle(qids)
    specs = [(qid, 0, []) for qid in qids[:args.limit]]
    verify_lock(out / "manifest.json", {"specs": specs, "seed": args.seed, "batch": args.batch, "max_tokens": args.max_tokens,
                                       "student": MODELS["sft"], "adapters": checked, "training_recipe": recipe})
    base, scores = collect(engine, store, Actor("sft", args.max_tokens), "dev", specs, out / "base", args.batch, 0, args.seed)
    if len(scores) != len(specs):
        raise RuntimeError("Dev reference has backend failures; resume before freezing fixed prefixes")
    prefixes = [(ep["qid"], 0, prefix) for ep, prefix in recovery_states(base, scores, failure_type)]
    verify_lock(out / "fixed-cohort.json", {"definition": failure_type, "specs": prefixes})
    paths = {"base": out / "base"}
    fixed_paths = {}
    for name, adapter in {"base": None, **adapters}.items():
        actor = Actor("sft", args.max_tokens, adapter=adapter)
        if name != "base":
            paths[name] = out / name
            collect(engine, store, actor, "dev", specs, paths[name], args.batch, 0, args.seed)
        fixed_paths[name] = out / f"fixed-{name}"
        if failure_type == "wrong_without_search":
            copy_initial_cohort(paths[name], fixed_paths[name], prefixes)
        else:
            collect(engine, store, actor, "dev", prefixes, fixed_paths[name], args.batch, 0, args.seed)
        del actor
    compare(paths, out)
    cohort_summary(paths, out, failure_type=failure_type)
    fixed_out = out / "fixed"
    fixed_out.mkdir(exist_ok=True)
    compare(fixed_paths, fixed_out)
    cohort_summary(fixed_paths, fixed_out, fixed_prefixes=True)
    engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline", "controls", "dev", "ablation-dev"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=["rl", "dev"], default="rl")
    parser.add_argument("--adapter-a", type=Path)
    parser.add_argument("--adapter-b", type=Path)
    parser.add_argument("--adapter-c", type=Path)
    parser.add_argument("--adapter-d", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if min(args.limit, args.batch, args.max_tokens) < 1:
        parser.error("--limit, --batch and --max-tokens must be positive")
    if args.stage in {"dev", "ablation-dev"}:
        if not args.adapter_a or not args.adapter_d:
            parser.error("Dev requires the two verified adapter directories")
        if args.stage == "ablation-dev" and (not args.adapter_b or not args.adapter_c):
            parser.error("Ablation dev evaluation requires all four adapter directories")
        evaluate_dev(args)
    elif args.stage == "baseline":
        out = args.run / f"{args.split}-comparison"
        out.mkdir(exist_ok=True)
        paths = {name: args.run / f"{args.split}-{name}" for name in ("base", "sft", "teacher")}
        compare(paths, out)
        review_baseline(paths, out, TaskStore(args.data), args.split, args.seed)
    else:
        ordinary_controls(args.run, args.run / "rl-recovery")


if __name__ == "__main__":
    main()
