"""Replay a fixed no-search routing policy on saved, paired model episodes."""
import argparse
import json
from pathlib import Path

from .demo_run import should_retry_with_directed_sft
from .download import sha256
from .reopd_compare import paired_interval


ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "local_dev": ROOT / "runs/reopd/formal-v2-expanded/ablation-dev-12000tokens-42-wrong-without-search",
    "local_test": ROOT / "runs/reopd/formal-v2-expanded/final-test",
    "external_hotpot": ROOT / "runs/reopd/public-eval-v1/hotpot-first",
    "external_2wiki": ROOT / "runs/reopd/public-eval-v1/reopd-2wiki",
}


def load_rows(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    table = {(row["qid"], row["attempt"]): row for row in records}
    if len(table) != len(records):
        raise ValueError(f"Duplicate episode keys: {path}")
    return table


def evaluate_cohort(directory):
    tables = {}
    hashes = {}
    for arm in ("base", "A"):
        episodes_path = directory / arm / "episodes.jsonl"
        scores_path = directory / arm / "scores.jsonl"
        episodes, scores = load_rows(episodes_path), load_rows(scores_path)
        if episodes.keys() != scores.keys():
            raise ValueError(f"Unscored episodes in {directory / arm}")
        tables[arm] = (episodes, scores)
        hashes[arm] = {"episodes": sha256(episodes_path), "scores": sha256(scores_path)}
    base_episodes, base_scores = tables["base"]
    _, directed_scores = tables["A"]
    if base_scores.keys() != directed_scores.keys():
        raise ValueError(f"Unpaired model cohorts: {directory}")
    chosen = {}
    routed = 0
    for key, episode in base_episodes.items():
        retry = should_retry_with_directed_sft(episode)
        routed += retry
        chosen[key] = directed_scores[key] if retry else base_scores[key]
    keys = sorted(chosen)
    def summary(scores):
        return {"supported_successes": sum(bool(scores[key]["supported_success"]) for key in keys),
                "answer_em_correct": sum(float(scores[key]["answer_em"]) for key in keys)}
    return {"questions": len(keys), "retried": routed,
            "base": summary(base_scores), "directed_sft": summary(directed_scores), "routed": summary(chosen),
            "routed_minus_base": {
                metric: paired_interval([(base_scores[key], chosen[key]) for key in keys], metric)
                for metric in ("supported_success", "answer_em")},
            "source_sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reopd/routed-eval/report.json")
    args = parser.parse_args()
    report = {"policy": "If original SFT finishes with zero searches, rerun from the question with directed SFT A; otherwise retain original SFT.",
              "source": "Saved full model episodes; policy uses only the original model's visible finish status and search count.",
              "cohorts": {name: evaluate_cohort(path) for name, path in COHORTS.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
