"""Measure real SFT sequence lengths without truncating or loading model weights."""
import argparse
import json
from pathlib import Path

from .agent import MODEL_ID, REVISION
from .expand_sft import read_jsonl
from .filings import ROOT
from .prepare_sft import sha256
from .trajectory_sources import OUTPUT

SNAPSHOT = ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots" / REVISION


def inspect(data=OUTPUT, snapshot=SNAPSHOT):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    report = {"model": MODEL_ID, "revision": REVISION, "dataset_manifest_sha256": sha256(data / "manifest.json"),
              "required_loss_scope": "completion only; mask the prompt, including earlier assistant actions",
              "splits": {}}
    for split in ("train", "validation", "test"):
        path = data / f"{split}-actions.jsonl"
        if sha256(path) != manifest["files_sha256"][path.name]:
            raise ValueError(f"Dataset file changed: {path}")
        rows, lengths, completions = read_jsonl(path), [], []
        for row in rows:
            rendered = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prefix = tokenizer(rendered, return_attention_mask=False).input_ids
            target = tokenizer(row["completion"][0]["content"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            lengths.append(len(prefix) + len(target))
            completions.append(len(target))
        lengths.sort()
        report["splits"][split] = {"rows": len(rows), "sequence_min": min(lengths),
            "sequence_median": lengths[len(lengths) // 2], "sequence_p95": lengths[int(len(lengths) * .95)],
            "sequence_max": max(lengths), "completion_max": max(completions),
            "completion_tokens_total": sum(completions), "samples_over_8192": sum(n > 8192 for n in lengths)}
    (data / "token-statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    inspect(args.data.resolve(), args.snapshot.resolve())
