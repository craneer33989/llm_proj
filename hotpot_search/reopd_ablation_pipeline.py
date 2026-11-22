"""Check and run the four-arm, two-seed experiment without opening held-out data."""
import argparse
import json
from pathlib import Path

from .reopd_baseline_pipeline import execute
from .reopd_compare import validate_adapters
from .reopd_final import ARMS, arm_paths
from .reopd_model import checkpoint
from .reopd_train import experiment_schedule


def prepare(run, budget, seeds, failure_type):
    lock = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    for model in ("sft", "teacher"):
        checkpoint(model)
    plan = []
    for seed in sorted(seeds):
        adapters = arm_paths(run, budget, seed, failure_type)
        arms = []
        for arm, adapter in adapters.items():
            schedule, manifest_hash = experiment_schedule(run, arm, seed, lock, failure_type)
            completed = adapter.parent.exists()
            if completed:
                recipe, _ = validate_adapters({arm: adapter}, lock, ablation=True)
                if (recipe["update_token_budget"], recipe["seed"], recipe["failure_type"],
                        recipe["ablation_pool_manifest_sha256"]) != (budget, seed, failure_type, manifest_hash):
                    raise ValueError(f"Existing training output has a different recipe: {adapter.parent}")
            arms.append({"arm": arm, "states": len(schedule), "adapter": str(adapter), "completed": completed})
        plan.append({"seed": seed, "arms": arms})
    return {"failure_type": failure_type, "update_tokens_per_arm": budget,
            "total_update_tokens": budget * len(ARMS) * len(seeds), "dev_questions_per_seed": 100, "runs": plan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--failure-type", choices=["early_stop", "wrong_without_search"], required=True)
    parser.add_argument("--update-tokens", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs=2, default=[42, 43])
    parser.add_argument("--confirm-p4-budget", action="store_true")
    args = parser.parse_args()
    if args.update_tokens < 1 or len(set(args.seeds)) != 2:
        parser.error("Require a positive token budget and two distinct training seeds")
    plan = prepare(args.run, args.update_tokens, args.seeds, args.failure_type)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.confirm_p4_budget:
        return
    for run in plan["runs"]:
        options = []
        for arm in run["arms"]:
            label = arm["arm"]
            if not arm["completed"]:
                execute("reopd_train", label, "--run", args.run, "--seed", run["seed"], "--failure-type", args.failure_type,
                        "--update-tokens", args.update_tokens, "--confirm-p4-budget")
            options.extend((f"--adapter-{label.lower()}", arm["adapter"]))
        execute("reopd_compare", "ablation-dev", "--run", args.run, "--limit", 100, *options)


if __name__ == "__main__":
    main()
