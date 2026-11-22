"""Run one unscored, user-supplied question through the existing ReOPD protocol."""
import argparse
import gc
import json
import sys
from pathlib import Path

from .reopd import Rollout
from .reopd_model import Actor
from .search import SearchEngine


def run_question(engine, actor, question):
    rollout = Rollout(engine, {"id": "interactive", "question": question}, "interactive")
    while not rollout.reason:
        rollout.advance(actor.generate([rollout.messages], 0)[0])
    steps = []
    for step in rollout.steps:
        visible = {key: step[key] for key in ("kind", "value", "raw", "observation", "doc_ids") if key in step}
        steps.append(visible)
    return {"question": question, "answer": rollout.env.answer or None, "reason": rollout.reason,
            "calls": len(rollout.env.session.trace), "steps": steps}


def should_retry_with_directed_sft(result):
    """Route using the visible SFT result only."""
    return result["reason"] == "finished" and result["calls"] == 0


def check_resources():
    import psutil
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Live 4B inference requires a CUDA GPU; use saved trajectory comparison instead")
    free_vram, _ = torch.cuda.mem_get_info()
    free_ram = psutil.virtual_memory().available
    gib = 1024 ** 3
    if free_vram < 12 * gib or free_ram < 8 * gib:
        raise RuntimeError(
            f"Insufficient free memory for 4B inference: GPU {free_vram / gib:.1f} GiB, "
            f"RAM {free_ram / gib:.1f} GiB; need at least 12 GiB GPU and 8 GiB RAM"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("base", "sft", "teacher", "A", "B", "C", "D", "route"), required=True)
    args = parser.parse_args()
    question = json.load(sys.stdin)["question"]
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 500:
        raise ValueError("Question must be 1–500 characters")
    check_resources()
    adapter = None
    model = args.model
    if model in "ABCD":
        adapter = (Path(__file__).resolve().parents[1] / "runs" / "reopd" / "formal-v2-expanded" /
                   f"ablation-{model}-12000tokens-42-wrong-without-search" / "adapter")
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Local adapter is missing: {model}")
        model = "sft"
    engine = SearchEngine()
    try:
        actor = Actor("sft" if model == "route" else model, max_tokens=512, adapter=adapter, direct_device_load=True)
        result = run_question(engine, actor, question)
        if model == "route":
            result["route"] = "original_sft"
            if should_retry_with_directed_sft(result):
                first_answer = result["answer"]
                del actor
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                directed = (Path(__file__).resolve().parents[1] / "runs" / "reopd" / "formal-v2-expanded" /
                            "ablation-A-12000tokens-42-wrong-without-search" / "adapter")
                if not (directed / "adapter_config.json").is_file():
                    raise FileNotFoundError(f"Local directed SFT adapter is missing: {directed}")
                actor = Actor("sft", max_tokens=512, adapter=directed, direct_device_load=True)
                result = run_question(engine, actor, question)
                result["route"] = "directed_sft_after_no_search"
                result["first_answer"] = first_answer
        print(json.dumps(result, ensure_ascii=False))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
