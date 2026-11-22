"""Measure a teacher release with the pinned student tokenizer, without weights."""
import argparse
from pathlib import Path

from .deepseek_trajectories import atomic_json, read_jsonl
from .inspect_trajectories import SNAPSHOT
from .prepare_sft import sha256


def inspect(release, snapshot):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    path = release / "train-actions.jsonl"
    lengths, targets = [], []
    for row in read_jsonl(path):
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer(prompt, add_special_tokens=False).input_ids
        completion = tokenizer(row["completion"][0]["content"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        lengths.append(len(prefix) + len(completion))
        targets.append(len(completion))
    if not lengths:
        raise ValueError("No teacher SFT actions to inspect")
    lengths.sort()
    targets.sort()
    report = {"student_tokenizer": str(snapshot), "train_actions_sha256": sha256(path), "rows": len(lengths),
              "thinking_enabled_for_student": False, "loss_scope": "completion and EOS only; mask all prompt tokens",
              "sequence_min": min(lengths), "sequence_median": lengths[len(lengths) // 2],
              "sequence_p95": lengths[min(len(lengths) - 1, int(len(lengths) * .95))], "sequence_max": max(lengths),
              "completion_median": targets[len(targets) // 2], "completion_max": max(targets),
              "completion_tokens_total": sum(targets), "sequence_tokens_total": sum(lengths),
              "samples_over_8192": sum(n > 8192 for n in lengths), "completions_over_384": sum(n > 384 for n in targets)}
    atomic_json(release / "token-statistics.json", report)
    return report


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    print(json.dumps(inspect(args.release.resolve(), args.snapshot.resolve()), ensure_ascii=False, indent=2))
