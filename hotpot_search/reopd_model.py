"""Pinned local checkpoint loading and batched one-action generation."""
import os
import json
from pathlib import Path

from .reopd import render

MODELS = {
    "base": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
    "sft": ("baohao/Search_Qwen3-4B-Instruct-2507_SFT", "19926d55956eaeb04775c89637a8948d750c3c80"),
    "teacher": ("baohao/Search_Qwen3-4B-Instruct-2507_SFT-RL", "041b174e3a9e5f37e85a4ec2426b2fd9830ac13e"),
    "diagnostic": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
}
CACHE = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", ".cache/huggingface")) / "hub"))


def verified_dir(name):
    return CACHE.parent / "reopd-verified" / f"{name}-{MODELS[name][1]}"


def checkpoint(name):
    from huggingface_hub import snapshot_download

    model_id, revision = MODELS[name]
    verified = verified_dir(name)
    marker = verified / "verified.json"
    if marker.is_file():
        info = json.loads(marker.read_text(encoding="utf-8"))
        if (info["model"], info["revision"]) != (model_id, revision) or any(
            not (verified / filename).is_file() or (verified / filename).stat().st_size != asset["size"]
            for filename, asset in info["files"].items()
        ):
            raise ValueError(f"Invalid verified checkpoint cache: {verified}")
        location = verified
    else:
        location = Path(snapshot_download(model_id, revision=revision, cache_dir=CACHE, local_files_only=True))
    index = location / "model.safetensors.index.json"
    weights = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if missing := sorted(filename for filename in weights if not (location / filename).is_file()):
        raise FileNotFoundError(f"Incomplete pinned checkpoint {name}: {', '.join(missing)}")
    return location


class Actor:
    def __init__(self, name, max_tokens=512, adapter=None, device=None, direct_device_load=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = checkpoint(name)
        self.tokenizer = AutoTokenizer.from_pretrained(source, padding_side="left", local_files_only=True)
        if direct_device_load:
            if device is not None or not torch.cuda.is_available():
                raise RuntimeError("Direct model loading requires an available CUDA GPU")
            self.model = AutoModelForCausalLM.from_pretrained(
                source, dtype=torch.bfloat16, local_files_only=True,
                device_map="cuda:0", low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16, local_files_only=True)
            self.model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_tokens = max_tokens

    def sample(self, histories, temperature=0.0):
        import torch

        batch = self.tokenizer([render(history) for history in histories], add_special_tokens=False,
                               padding=True, return_tensors="pt").to(self.model.device)
        options = {"do_sample": bool(temperature), "temperature": temperature or 1.0, "top_p": 1.0, "top_k": 0}
        with torch.inference_mode():
            generated = self.model.generate(**batch, max_new_tokens=self.max_tokens,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            stop_strings=["</tool_call>"], tokenizer=self.tokenizer, **options)
        return batch, generated[:, batch.input_ids.shape[1]:]

    def generate(self, histories, temperature=0.0):
        _, continuation = self.sample(histories, temperature)
        return self.tokenizer.batch_decode(continuation, skip_special_tokens=True)
