"""Run formal SFT smoke and the three pinned RL baselines in gate order."""
import argparse
from pathlib import Path
import subprocess
import sys

from .reopd_model import checkpoint
from .search import DEFAULT_DATA


def execute(module, *arguments):
    subprocess.run([sys.executable, "-u", "-m", f"hotpot_search.{module}", *map(str, arguments)], check=True)


def pipeline(data, smoke, run, batch, max_tokens, seed):
    checkpoint("sft")
    execute("reopd_run", "reference", "--data", data, "--run", smoke)
    execute("reopd_run", "baseline", "--data", data, "--run", smoke, "--model", "sft", "--split", "dev", "--limit", 10,
            "--batch", batch, "--max-tokens", max_tokens, "--seed", seed)
    execute("reopd_preflight", "--data", data, "--run", smoke)
    for model in ("base", "sft", "teacher"):
        checkpoint(model)
    for model in ("base", "sft", "teacher"):
        execute("reopd_run", "baseline", "--data", data, "--run", run, "--model", model, "--split", "rl", "--limit", 100,
                "--batch", batch, "--max-tokens", max_tokens, "--seed", seed)
    execute("reopd_compare", "baseline", "--data", data, "--run", run, "--seed", seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch, args.max_tokens) < 1 or args.run.resolve() == args.smoke.resolve():
        parser.error("Use positive budgets and distinct smoke and baseline directories")
    pipeline(args.data, args.smoke, args.run, args.batch, args.max_tokens, args.seed)


if __name__ == "__main__":
    main()
