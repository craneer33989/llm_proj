"""Run one unscored, user-supplied question through the existing ReOPD protocol."""
import argparse
import gc
import json
import sys
from pathlib import Path

from .reopd import Rollout
from .reopd_model import Actor, checkpoint
from .search import SearchEngine

ROOT = Path(__file__).resolve().parents[1]
RESIDENT_MODELS = ("sft", "A", "route")
MODELS = ("base", "sft", "teacher", "A", "B", "C", "D", "route")


class DemoActor(Actor):
    def __init__(self, name, max_tokens=512, adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Direct model loading requires an available CUDA GPU")
        source = checkpoint(name)
        self.tokenizer = AutoTokenizer.from_pretrained(source, padding_side="left", local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, local_files_only=True,
            device_map="cuda:0", low_cpu_mem_usage=True,
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_tokens = max_tokens


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


def validate_question(question):
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 500:
        raise ValueError("Question must be 1–500 characters")


def adapter_path(arm):
    path = ROOT / "runs/reopd/formal-v2-expanded" / f"ablation-{arm}-12000tokens-42-wrong-without-search/adapter"
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Local adapter is missing: {arm}")
    return path


def routed_result(first, retry):
    if should_retry_with_directed_sft(first):
        return {**retry(), "route": "directed_sft_after_no_search", "first_answer": first["answer"]}
    return {**first, "route": "original_sft"}


class ResidentSearch:
    """Keep one search SFT checkpoint with adapter A. Callers serialize access."""

    def __init__(self):
        self.actor = None

    def load(self):
        if self.actor is None:
            check_resources()
            self.actor = DemoActor("sft", adapter=adapter_path("A"))

    def release(self):
        if self.actor is not None:
            import torch

            self.actor = None
            gc.collect()
            torch.cuda.empty_cache()

    def run(self, engine, model, question):
        if model == "A":
            return run_question(engine, self.actor, question)
        with self.actor.model.disable_adapter():
            first = run_question(engine, self.actor, question)
        if model == "sft":
            return first
        return routed_result(first, lambda: run_question(engine, self.actor, question))


def run_child_route(engine, question):
    import torch

    actor = DemoActor("sft")
    first = run_question(engine, actor, question)

    def retry():
        nonlocal actor
        actor = None
        gc.collect()
        torch.cuda.empty_cache()
        directed = DemoActor("sft", adapter=adapter_path("A"))
        return run_question(engine, directed, question)

    return routed_result(first, retry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    model = parser.parse_args().model
    question = json.load(sys.stdin)["question"]
    validate_question(question)
    check_resources()
    engine = SearchEngine()
    try:
        if model == "route":
            result = run_child_route(engine, question)
        else:
            adapter = adapter_path(model) if model in ("A", "B", "C", "D") else None
            actor = DemoActor("sft" if adapter else model, adapter=adapter)
            result = run_question(engine, actor, question)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
