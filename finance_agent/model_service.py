"""Serve the pinned Qwen checkpoint separately from the data application."""
from contextlib import asynccontextmanager
import os
from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LOCK = Lock()


@asynccontextmanager
async def lifespan(app):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    source = os.getenv("FINANCE_MODEL_PATH", MODEL_ID)
    app.state.tokenizer = AutoTokenizer.from_pretrained(source, revision=REVISION, local_files_only=True)
    app.state.model = AutoModelForCausalLM.from_pretrained(
        source, revision=REVISION, local_files_only=True, dtype=dtype,
    ).to(device).eval()
    app.state.device = device
    yield


app = FastAPI(lifespan=lifespan)


class Prompt(BaseModel):
    messages: list[dict[str, str]] = Field(min_length=1, max_length=8)


@app.get("/health")
def health():
    return {"device": app.state.device, "model": MODEL_ID, "revision": REVISION}


@app.post("/generate")
def generate(payload: Prompt):
    with LOCK:
        return {"text": generate_text(app.state.model, app.state.tokenizer, payload.messages, app.state.device,
                                      max_new_tokens=96)}


def encode_prompt(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(rendered, return_tensors="pt")


@torch.inference_mode()
def generate_text(model, tokenizer, messages, device, max_new_tokens=48):
    inputs = encode_prompt(tokenizer, messages).to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id, stop_strings=["}"], tokenizer=tokenizer)
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
