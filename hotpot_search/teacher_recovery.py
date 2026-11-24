"""Collect Flash continuations from saved ReOPD early-stop training histories."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import urllib.error
import urllib.request

from .reopd import BackendError, Rollout, environment_lock
from .reopd_run import write_rows
from .search import DEFAULT_DATA, SearchEngine
from .teacher_select import ENDPOINT, BudgetStop, NoRedirect, ProviderError, atomic_json, load_section_key
from .training import TaskStore


FLASH_MODEL = "deepseek-flash"


class Teacher:
    def __init__(self, secret, output, budget_cny=5.0, model="deepseek-v4-pro", *,
                 thinking=True, max_tokens=4096):
        self.secret, self.output, self.limit, self.model = secret, Path(output), budget_cny, model
        self.thinking, self.max_tokens = thinking, max_tokens
        self.input_rate, self.output_rate = (2.0, 8.0) if model == FLASH_MODEL else (9.0, 27.0)
        self.opener = urllib.request.build_opener(NoRedirect())
        ledger = self.output / "usage.json"
        self.usage = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else {
            "requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "conservative_cny": 0.0,
            "pricing": {"input_cny_per_million": self.input_rate, "output_cny_per_million": self.output_rate,
                        "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
                        "note": "Peak uncached rate applied to all tokens; not exact billed cost"}}

    def complete(self, messages, tools, record):
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "thinking": {"type": "enabled" if self.thinking else "disabled"},
                   "max_tokens": self.max_tokens}
        if tools:
            payload["tools"] = tools
        if self.thinking:
            payload["reasoning_effort"] = "low"
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        # Bytes bound token count conservatively; overhead allowance for tool encoding.
        ceiling = ((len(encoded) + 8192) * self.input_rate + self.max_tokens * self.output_rate) / 1e6
        if self.usage["conservative_cny"] + ceiling > self.limit:
            raise BudgetStop("Conservative per-request reservation would exceed budget")
        self.usage["requests"] += 1
        self.usage["conservative_cny"] += ceiling
        atomic_json(self.output / "usage.json", self.usage)
        record({"event": "request", "payload": payload, "reserved_cny": ceiling})
        req = urllib.request.Request(ENDPOINT, data=encoded, headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.secret})
        try:
            with self.opener.open(req, timeout=180) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            record({"event": "provider_error", "status": status})
            raise ProviderError(status) from None
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            record({"event": "provider_error", "status": "network_or_invalid_json"})
            raise ProviderError("network_or_invalid_json") from None
        usage = result.get("usage", {})
        if isinstance(usage.get("prompt_tokens"), int) and isinstance(usage.get("completion_tokens"), int):
            prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
            actual_upper = (prompt * self.input_rate + completion * self.output_rate) / 1e6
            self.usage["prompt_tokens"] += prompt
            self.usage["completion_tokens"] += completion
            self.usage["conservative_cny"] += actual_upper - ceiling
        self.usage["conservative_cny"] = round(self.usage["conservative_cny"], 8)
        atomic_json(self.output / "usage.json", self.usage)
        record({"event": "response", "body": result})
        return result



EVIDENCE_INSTRUCTION = (
    "Base your final answer only on the retrieved passages. Before finishing, check that those "
    "passages establish each fact needed to answer the question. If a fact is missing, search for "
    "that fact using the most specific entity or clue already found, even if you recall the answer. "
    "Finish as soon as the retrieved evidence is sufficient."
)


def chat_messages(history, require_search=False):
    system, *turns = history
    messages = [{**system, "content": system["content"] + "\n\n" + EVIDENCE_INSTRUCTION}]
    if require_search:
        messages[0]["content"] += (
            "\nThis search history is missing evidence needed to answer the question. "
            "Your next action must be one search for the missing fact. Choose the query yourself."
        )
    for message in turns:
        if message["role"] == "tool":
            message = {"role": "user", "content": f"<tool_response>\n{message['content']}\n</tool_response>"}
        messages.append(message)
    return messages


def continue_episode(env, teacher, gold, force_first_search=False):
    response_meta = []
    prefix_size = len(env.steps)

    def record(event):
        if event["event"] == "response":
            body = event["body"]
            response_meta.append({"model": body.get("model"), "usage": body.get("usage")})

    while not env.reason:
        require_search = force_first_search and len(env.steps) == prefix_size
        response = teacher.complete(chat_messages(env.messages, require_search), None, record)
        choice = response["choices"][0]
        raw = choice["message"].get("content")
        if choice["finish_reason"] == "length":
            env.reason = "generation_truncated"
        else:
            env.advance(raw if isinstance(raw, str) else "")
    episode, score = env.record(gold, 0)
    return {"episode": episode, "score": score, "provider": response_meta,
            "review_candidate": bool(score["supported_success"] and episode["steps"][prefix_size]["kind"] == "search")}


def collect(args, engine, store, secret):
    run = args.source_run
    lock = environment_lock(engine)
    if lock != json.loads((run / "environment.json").read_text(encoding="utf-8")):
        raise ValueError("Local search environment differs from the saved prefixes")
    manifest = json.loads((run / "rl-recovery" / "manifest.json").read_text(encoding="utf-8"))
    if manifest["split"] != "rl" or manifest.get("failure_type", "early_stop") != "early_stop":
        raise ValueError("Expected the RL early-stop recovery manifest")
    prefixes = {qid: prefix for qid, attempt, prefix in manifest["specs"] if attempt == 0}
    rng = random.Random(args.seed)
    smoke = rng.sample(sorted(prefixes), min(5, len(prefixes)))
    remaining = sorted(prefixes.keys() - set(smoke))
    rng.shuffle(remaining)
    sample = smoke + remaining
    qids = sample[args.offset:args.offset + args.limit]
    if not qids:
        raise ValueError("No recovery questions at the requested offset")
    envs = []
    for qid in qids:
        prefix = prefixes[qid]
        if len(prefix) != 1 or prefix[0]["kind"] != "search":
            raise ValueError(f"Expected one saved search for {qid}")
        envs.append(Rollout(engine, store.task("rl", qid), "rl", prefix))

    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "config.json", {"model": FLASH_MODEL, "thinking": False, "max_tokens": 1024,
                "evidence_instruction": EVIDENCE_INSTRUCTION,
                "source_run": str(run), "qids": qids, "seed": args.seed, "offset": args.offset,
                "force_first_search": args.force_first_search,
                "budget_cny": args.budget_cny, "environment": lock})
    teacher = Teacher(secret, args.output, args.budget_cny, FLASH_MODEL, thinking=False, max_tokens=1024)
    records, stop = [], None
    for qid, env in zip(qids, envs, strict=True):
        try:
            result = continue_episode(env, teacher, store.gold["rl"][qid], args.force_first_search)
        except (BudgetStop, ProviderError) as error:
            stop = {"qid": qid, "reason": type(error).__name__, "status": getattr(error, "status", None)}
            break
        except BackendError:
            stop = {"qid": qid, "reason": "BackendError"}
            break
        records.append(result)
        write_rows(args.output / "episodes.jsonl", records)
        print(json.dumps({"qid": qid, "reason": result["episode"]["reason"],
                          "review_candidate": result["review_candidate"]}), flush=True)
    summary = {"selected": len(qids), "scored": len(records),
               "answer_em": sum(row["score"]["answer_em"] for row in records),
               "continued_search": sum(row["episode"]["calls"] > 1 for row in records),
               "full_evidence": sum(row["score"]["support_sentence_recall"] == 1 for row in records),
               "supported_success": sum(row["score"]["supported_success"] for row in records),
               "review_candidates": sum(row["review_candidate"] for row in records),
               "termination": dict(Counter(row["episode"]["reason"] for row in records)),
               "stop": stop, "usage": teacher.usage}
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return stop is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", default=DEFAULT_DATA, type=Path)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--force-first-search", action="store_true",
                        help="Tell Flash the saved prefix lacks evidence and require a search as its next action")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-cny", type=float, default=1.0)
    args = parser.parse_args()
    if args.limit < 1 or args.offset < 0 or not 0 < args.budget_cny < float("inf"):
        parser.error("limit and budget must be positive and finite, and offset nonnegative")
    secret = load_section_key(args.key_file) if args.key_file else os.environ.get("DEEPSEEK_API_KEY")
    if not secret:
        parser.error("set DEEPSEEK_API_KEY or pass --key-file")
    engine = SearchEngine(args.data)
    try:
        return 0 if collect(args, engine, TaskStore(args.data), secret) else 1
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
