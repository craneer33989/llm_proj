"""Freeze the dev experiment, then run one isolated held-out evaluation."""
import argparse
import json
from pathlib import Path

from .download import sha256
from .reopd import environment_lock
from .reopd_compare import cohort_summary, compare, copy_initial_cohort, validate_adapters
from .reopd_model import Actor, MODELS, checkpoint
from .reopd_run import collect, recovery_states, verify_lock, write_json
from .search import DEFAULT_DATA, SearchEngine, rows

ARMS = ("A", "B", "C", "D")
RULE = "one terminal checkpoint per arm at identical student token budget; lower training seed is primary"
REPORT_POLICY = {"min_fixed_distinct_questions": 20, "test_questions": 100,
                 "superiority": "paired 95% interval strictly above zero"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def code_identity():
    return {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def base_identity():
    directory = checkpoint("sft")
    return {path.name: sha256(path) for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".jinja"}}


def fault_suffix(failure_type):
    return "-wrong-without-search" if failure_type == "wrong_without_search" else ""


def arm_paths(run, budget, seed, failure_type="early_stop"):
    return {arm: run / f"ablation-{arm}-{budget}tokens-{seed}{fault_suffix(failure_type)}" / "adapter" for arm in ARMS}


def validated_dev(run, budget, seed, lock, failure_type="early_stop"):
    adapters = arm_paths(run, budget, seed, failure_type)
    recipe, hashes = validate_adapters(adapters, lock, ablation=True)
    if (recipe["seed"], recipe["update_token_budget"], recipe["failure_type"]) != (seed, budget, failure_type):
        raise ValueError("Checkpoint seed or budget differs from the requested dev evidence")
    directory = run / f"ablation-dev-{budget}tokens-{seed}{fault_suffix(failure_type)}"
    manifest = read_json(directory / "manifest.json")
    if manifest["adapters"] != hashes or manifest["training_recipe"] != recipe:
        raise ValueError("Dev did not evaluate the frozen adapter contents")
    paired = read_json(directory / "paired_metrics.json")
    fixed = read_json(directory / "fixed" / "cohort_metrics.json")
    counts = read_json(directory / "cohort_metrics.json")["arms"]
    if (len({spec[0] for spec in manifest["specs"]}) != 100 or paired["requested_pairs"] != 100
            or any(counts[arm]["scorable"] != 100 for arm in ("base", *ARMS))):
        raise ValueError("P5 requires all 100 matched dev episodes")
    fixed_counts = [fixed["arms"][arm] for arm in ("base", *ARMS)]
    if not fixed_counts[0]["requested"] or any(row["scorable"] != fixed_counts[0]["requested"] for row in fixed_counts):
        raise ValueError("Dev fixed-prefix cohort is empty or unresolved")
    files = ("manifest.json", "paired_metrics.json", "cohort_metrics.json", "fixed-cohort.json",
             "fixed/paired_metrics.json", "fixed/cohort_metrics.json")
    return manifest, {"seed": seed, "adapters": hashes, "recipe": recipe,
                      "dev_sha256": {filename: sha256(directory / filename) for filename in files}}


def frozen_plan(engine, run, budget, seeds, failure_type="early_stop"):
    if budget < 1 or len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("Freeze requires a positive budget and two distinct training seeds")
    lock = environment_lock(engine)
    if read_json(run / "environment.json") != lock:
        raise ValueError("Run and evaluator environment differ")
    primary, replicate = sorted(seeds)
    first, main = validated_dev(run, budget, primary, lock, failure_type)
    second, check = validated_dev(run, budget, replicate, lock, failure_type)
    settings = ("specs", "seed", "batch", "max_tokens", "student")
    if (any(first[key] != second[key] for key in settings)
            or main["dev_sha256"]["fixed-cohort.json"] != check["dev_sha256"]["fixed-cohort.json"]):
        raise ValueError("Dev seed replication used different evaluation settings")
    train_keys = ("max_tokens", "lr", "rank", "alpha", "pool_sha256", "audit_sha256")
    if any(main["recipe"][key] != check["recipe"][key] for key in train_keys):
        raise ValueError("Training seed replication changed shared training settings")
    return {"rule": RULE, "environment": lock, "code_sha256": code_identity(), "base_sha256": base_identity(),
            "student": list(MODELS["sft"]), "update_tokens": budget, "primary": main, "replicate": check,
            "evaluation": {key: first[key] for key in ("seed", "batch", "max_tokens")},
            "test_size": 100, "failure_type": failure_type, "fixed_prefix_rule": f"unadapted SFT: {failure_type}",
            "report_policy": REPORT_POLICY,
            "comparison": "paired case-cluster bootstrap; review estimates on test without retuning"}


class HeldOutStore:
    def __init__(self, data, expected):
        manifest = read_json(data / "manifest.json")
        files = ("test.jsonl", "private/test.gold.jsonl")
        loaded = []
        for filename in files:
            path = data / filename
            if sha256(path) != manifest["file_sha256"][filename]:
                raise ValueError(f"Held-out file hash differs: {filename}")
            records = list(rows(path))
            by_id = {record["id"]: record for record in records}
            if len(records) != expected or len(by_id) != expected:
                raise ValueError(f"Expected {expected} distinct held-out questions in {filename}")
            loaded.append(by_id)
        tasks, gold = loaded
        if tasks.keys() != gold.keys() or any(set(task) != {"id", "question"} for task in tasks.values()):
            raise ValueError("Held-out public tasks and private scores do not align")
        self.tasks, self.gold = {"test": tasks}, {"test": gold}


def interpret_results(fixed, fixed_pairs, full, full_pairs, policy):
    fixed_base, full_base = fixed["arms"]["base"], full["arms"]["base"]
    fixed_total = fixed_base["requested"]
    if (fixed_base["distinct_questions"] < policy["min_fixed_distinct_questions"]
            or any(row["scorable"] != fixed_total for row in fixed["arms"].values())):
        return "insufficient_fixed_evidence"
    if (full_base["requested"] != policy["test_questions"]
            or any(row["scorable"] != full_base["requested"] for row in full["arms"].values())):
        return "insufficient_end_to_end_evidence"

    def higher(results, arm, count):
        contrast = results["comparisons"][f"D_minus_{arm}"]["supported_success"]
        interval = contrast["interval_95"]
        return contrast["paired_prefixes"] == count and interval is not None and interval[0] > 0

    if not higher(fixed_pairs, "B", fixed_total):
        return "no_clear_recovery_advantage_over_teacher_replay"
    if not higher(fixed_pairs, "A", fixed_total):
        return "prefer_directed_sft_pending_evidence_over_a"
    if not higher(fixed_pairs, "C", fixed_total):
        return "no_clear_advantage_over_untargeted_mixture"
    if not higher(full_pairs, "A", full_base["requested"]):
        return "fixed_prefix_gain_without_clear_end_to_end_gain"
    return "directed_reopd_supported_on_frozen_cohort"


def write_report(out, plan, adapters):
    full, fixed = (read_json(directory / "cohort_metrics.json") for directory in (out, out / "fixed"))
    full_pairs, fixed_pairs = (read_json(directory / "paired_metrics.json") for directory in (out, out / "fixed"))
    verdict = interpret_results(fixed, fixed_pairs, full, full_pairs, plan["report_policy"])
    costs = {arm: read_json(path.parent / "metrics.json") for arm, path in adapters.items()}
    descriptions = {
        "insufficient_fixed_evidence": "固定前缀可评估样本不足，不作恢复效果结论。",
        "insufficient_end_to_end_evidence": "端到端队列未全量可评分，不作总体优效结论。",
        "no_clear_recovery_advantage_over_teacher_replay": "没有证据证明 D 的固定前缀恢复优于 B。",
        "prefer_directed_sft_pending_evidence_over_a": "D 的固定前缀恢复优于 B，但未证明优于 A；优先采用定向 SFT。",
        "no_clear_advantage_over_untargeted_mixture": "没有证据证明定向筛选优于同配比的普通混合。",
        "fixed_prefix_gain_without_clear_end_to_end_gain": "固定前缀有改善，端到端相对 A 未证明有收益；先核查自然早停覆盖率。",
        "directed_reopd_supported_on_frozen_cohort": "冻结队列支持 D 的固定前缀改善及相对 A 的端到端改善，不外推到其他索引。",
    }
    lines = ["# HotpotQA ReOPD 最终评测", "", descriptions[verdict], "",
             f"主训练种子：{plan['primary']['seed']}；dev 复核种子：{plan['replicate']['seed']}。",
             f"未适配 SFT 的目标故障（{plan['failure_type']}）：{full['natural_target_failure_count']}/{full['natural_scorable_count']}。",
             "", "| 组 | 端到端支持成功 | 固定前缀支持成功 | 简单题回退/可比较 | 学生更新 token | 教师打分输入 token | 训练墙钟 s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("base", *ARMS):
        end, prefix = full["arms"][name], fixed["arms"][name]
        cost = costs.get(name, {})
        lines.append(f"| {name} | {end['supported_successes']}/{end['requested']}（未判定 {end['unresolved']}） | "
                     f"{prefix['supported_successes']}/{prefix['requested']}（未判定 {prefix['unresolved']}） | "
                     f"{end['easy_regressions']}/{end['easy_paired']} | {cost.get('student_update_tokens', '—')} | "
                     f"{cost.get('teacher_scoring_input_tokens', '—')} | {round(cost['seconds'], 2) if cost else '—'} |")
    lines.extend(["", "| 对比（支持成功率差） | 固定前缀 95% 区间 | 端到端 95% 区间 |", "|---|---:|---:|"])
    for arm in ("A", "B", "C"):
        intervals = []
        for results in (fixed_pairs, full_pairs):
            contrast = results["comparisons"][f"D_minus_{arm}"]["supported_success"]
            interval = contrast["interval_95"]
            intervals.append("—" if interval is None else
                             f"{contrast['delta']:+.1%} [{interval[0]:+.1%}, {interval[1]:+.1%}]")
        lines.append(f"| D − {arm} | {intervals[0]} | {intervals[1]} |")
    lines.extend(["", "逐题得失和调用数见同目录配对与队列产物；没有证明优效不等于证明等效。",
                  "本地交付为 HotpotQA/BM25 环境、前缀筛选、LoRA 训练与评估。SFT 与 GRPO 教师初始化来自上游公开权重；未本地重训上游 GRPO。",
                  "本地是受限语料实验，不是官方 fullwiki 分数；公开权重与本地问题的潜在重合未排除。"])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "conclusion.json", {"verdict": verdict, "report_policy": plan["report_policy"], "training_costs": costs})


def evaluate(engine, data, run, plan):
    from filelock import FileLock

    out = run / "final-test"
    out.mkdir(exist_ok=True)
    with FileLock(str(out / ".evaluation.lock"), timeout=0):
        receipt = out / "complete.json"
        plan_path = run / "test-freeze.json"
        verify_lock(out / "started.json", {"freeze_sha256": sha256(plan_path)})
        if receipt.is_file():
            raise ValueError("Held-out evaluation already completed; use the existing result")
        store = HeldOutStore(data, plan["test_size"])
        options = plan["evaluation"]
        specs = [(qid, 0, []) for qid in sorted(store.tasks["test"])]
        failure_type = plan["failure_type"]
        initial = failure_type == "wrong_without_search"
        adapters = arm_paths(run, plan["update_tokens"], plan["primary"]["seed"], failure_type)
        paths = {name: out / name for name in ("base", *ARMS)}
        fixed_paths = {name: out / "fixed" / name for name in paths}
        base = Actor("sft", options["max_tokens"])
        episodes, scores = collect(engine, store, base, "test", specs, paths["base"], options["batch"], 0, options["seed"])
        if len(scores) != len(specs):
            raise RuntimeError("Held-out reference has backend failures; resume the same frozen run")
        prefixes = [(episode["qid"], 0, prefix) for episode, prefix in recovery_states(episodes, scores, failure_type)]
        verify_lock(out / "fixed-cohort.json", {"specs": prefixes, "rule": plan["fixed_prefix_rule"]})
        if initial:
            copy_initial_cohort(paths["base"], fixed_paths["base"], prefixes)
        else:
            collect(engine, store, base, "test", prefixes, fixed_paths["base"], options["batch"], 0, options["seed"])
        del base
        for name, adapter in adapters.items():
            actor = Actor("sft", options["max_tokens"], adapter=adapter)
            collect(engine, store, actor, "test", specs, paths[name], options["batch"], 0, options["seed"])
            if initial:
                copy_initial_cohort(paths[name], fixed_paths[name], prefixes)
            else:
                collect(engine, store, actor, "test", prefixes, fixed_paths[name], options["batch"], 0, options["seed"])
            del actor
        compare(paths, out)
        cohort_summary(paths, out, failure_type=failure_type)
        compare(fixed_paths, out / "fixed")
        cohort_summary(fixed_paths, out / "fixed", fixed_prefixes=True)
        write_report(out, plan, adapters)
        write_json(receipt, {"freeze_sha256": sha256(plan_path), "test_questions": len(specs),
                            "fixed_prefixes": len(prefixes), "results_sha256": {name: sha256(out / name) for name in
                            ("report.md", "conclusion.json", "paired_results.csv", "paired_metrics.json", "cohort_metrics.json", "changed_cases.jsonl",
                             "fixed/paired_results.csv", "fixed/paired_metrics.json", "fixed/cohort_metrics.json",
                             "fixed/changed_cases.jsonl")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "evaluate"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--update-tokens", type=int)
    parser.add_argument("--seeds", type=int, nargs=2)
    parser.add_argument("--failure-type", choices=["early_stop", "wrong_without_search"], default="early_stop")
    parser.add_argument("--confirm-held-out", action="store_true")
    args = parser.parse_args()
    path = args.run / "test-freeze.json"
    if args.stage == "freeze" and (args.update_tokens is None or args.seeds is None):
        parser.error("Freeze requires --update-tokens and --seeds")
    if args.stage == "evaluate" and not args.confirm_held_out:
        parser.error("Use --confirm-held-out only after reviewing the frozen plan")
    engine = SearchEngine(args.data)
    try:
        if args.stage == "freeze":
            verify_lock(path, frozen_plan(engine, args.run, args.update_tokens, args.seeds, args.failure_type))
        else:
            plan = read_json(path)
            seeds = [plan[key]["seed"] for key in ("primary", "replicate")]
            if frozen_plan(engine, args.run, plan["update_tokens"], seeds, plan["failure_type"]) != plan:
                raise ValueError("Frozen plan changed; held-out files were not opened")
            evaluate(engine, args.data, args.run, plan)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
