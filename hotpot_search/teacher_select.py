"""Blind DeepSeek tool rollouts and conservative multi-hop subset selection.

Reads only the key immediately below the requested section. No key is copied,
printed, sent to the local service, or stored in dataset/config/request logs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from .client import SearchClient
from .multihop import coverage, dependency_links
from .prepare import write_jsonl
from .search import DEFAULT_DATA, rows
from .training import TaskStore

ENDPOINT = "https://api.deepseek.com/chat/completions"


def load_section_key(path, section="sft data"):
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Cannot decode credential file")
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if re.fullmatch(r"\s*" + r"\s*".join(map(re.escape, section.split())) + r"\s*[:：]?\s*", line, re.I)]
    if len(headers) != 1:
        raise ValueError("Expected exactly one requested credential section; no contents logged")
    value = next((line.strip() for line in lines[headers[0]+1:] if line.strip()), "")
    if not re.fullmatch(r"sk-[A-Za-z0-9_-]{16,}", value):
        raise ValueError("The next nonempty line must contain exactly one API key; no contents logged")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.replace(path)


class BudgetStop(Exception):
    pass


class ProviderError(Exception):
    def __init__(self, status):
        super().__init__(f"DeepSeek request failed (status={status}); response headers/body not logged")
        self.status = status


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Teacher:
    def __init__(self, secret, output, budget_cny=5.0, model="deepseek-v4-pro"):
        self.secret, self.output, self.limit, self.model = secret, Path(output), budget_cny, model
        self.opener = urllib.request.build_opener(NoRedirect())
        ledger = self.output / "usage.json"
        self.usage = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else {
            "requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "conservative_cny": 0.0,
            "pricing": {"input_cny_per_million": 9.0, "output_cny_per_million": 27.0,
                        "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
                        "note": "Peak uncached Pro rate applied to all tokens; not exact billed cost"}}

    def complete(self, messages, tools, record):
        payload = {"model": self.model, "messages": messages, "tools": tools, "stream": False,
                   "thinking": {"type": "enabled"}, "reasoning_effort": "low", "max_tokens": 4096}
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        # Bytes bound token count conservatively; overhead allowance for tool encoding.
        ceiling = ((len(encoded) + 8192) * 9 + 4096 * 27) / 1e6
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
            actual_upper = (prompt * 9 + completion * 27) / 1e6
            self.usage["prompt_tokens"] += prompt
            self.usage["completion_tokens"] += completion
            self.usage["conservative_cny"] += actual_upper - ceiling
        self.usage["conservative_cny"] = round(self.usage["conservative_cny"], 8)
        atomic_json(self.output / "usage.json", self.usage)
        record({"event": "response", "body": result})
        return result


def local_request(base, path, body=None, method=None):
    content = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=content, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def export(output):
    accepted, records = [], []
    for path in sorted((output / "episodes").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        records.append(result)
        if result.get("accepted"):
            accepted.append(result)
    write_jsonl(output / "selected_questions.jsonl", [{"id": r["id"], "question": r["question"]} for r in accepted])
    write_jsonl(output / "teacher_sft.jsonl", [{"id": r["id"], "messages": r["sft_messages"], "tools": r["tools"]} for r in accepted])
    write_jsonl(output / "selection_audit.jsonl", [{k: v for k, v in r.items() if k not in ("sft_messages", "tools", "events")} for r in records])
    usage = json.loads((output / "usage.json").read_text(encoding="utf-8")) if (output / "usage.json").exists() else {}
    summary = {"attempted": len(records), "accepted": len(accepted),
               "rejection_reasons": dict(Counter(reason for r in records for reason in r.get("rejections", []))),
               "usage": usage, "model": "deepseek-v4-pro", "gold_visible_to_teacher": False,
               "scope": "sft_seed only; local BM25 top3 per call; max6 calls",
               "claim": "Verified observed multi-step entity dependencies; NOT a proof that all possible one-shot queries fail"}
    atomic_json(output / "summary.json", summary)
    return summary


def run_episode(candidate, teacher, store, search, output, training_url):
    rid = candidate["id"]
    event_path = output / "provider_events" / f"{rid}.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    def record(value):
        serialized = json.dumps(value, ensure_ascii=False).replace(teacher.secret, "[REDACTED]")
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")

    reset = local_request(training_url, "/reset", {"split": "sft_seed", "task_id": rid, "mode": "bm25",
                            "max_calls": 6, "max_top_k": 3, "run_label": "teacher-deepseek-multihop-selection"})
    eid, tools, messages = reset["episode_id"], reset["tools"], reset["messages"]
    result = {"id": rid, "question": candidate["question"], "episode_id": eid, "events": [],
              "tools": tools, "gold_visible": False, "accepted": False, "rejections": [],
              "started_at": datetime.now(timezone.utc).isoformat(), "original_query_baseline": candidate}
    terminal = False
    try:
        for turn in range(1, 9):
            response = teacher.complete(messages, tools, record)
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                result["rejections"].append("generation_truncated")
                break
            raw = choice["message"]
            assistant = {k: raw[k] for k in ("role", "content", "tool_calls", "reasoning_content") if k in raw}
            assistant.setdefault("role", "assistant")
            calls = assistant.get("tool_calls") or []
            if not calls:
                result["rejections"].append("no_finish_tool")
                break
            messages.append(assistant)
            for call_index, call in enumerate(calls):
                function = call["function"]
                try:
                    arguments = json.loads(function["arguments"])
                except (ValueError, TypeError):
                    result["rejections"].append("invalid_tool_json")
                    terminal = True
                    break
                step = local_request(training_url, f"/episodes/{eid}/step", {"name": function["name"], "arguments": arguments})
                observation = step["observation"]
                result["events"].append({"name": function["name"], "arguments": arguments,
                                         "observation": observation, "provider_turn": turn})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(observation, ensure_ascii=False)})
                if observation.get("retryable"):
                    result["rejections"].append("search_backend_error")
                    terminal = True
                    break
                if step["terminated"] or step["truncated"]:
                    result["metrics"] = step["info"].get("metrics", {})
                    result["truncated"] = step["truncated"]
                    terminal = True
                    if call_index != len(calls) - 1:
                        result["rejections"].append("incomplete_tool_batch")
                    break
            if terminal:
                break
        else:
            result["rejections"].append("model_turn_limit")
    except BudgetStop:
        result["rejections"].append("budget_stop")
        raise
    except ProviderError as error:
        result["rejections"].append("provider_error")
        result["provider_status"] = error.status
        raise
    finally:
        result["sft_messages"] = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
        gold = store.gold["sft_seed"][rid]
        searches = [e for e in result["events"] if e["name"] == "local_search"]
        links = dependency_links(candidate["question"], gold, result["events"])
        result["dependency_links"] = links
        result["search_calls"] = len(searches)
        if not result.get("metrics", {}).get("supported_success"):
            result["rejections"].append("not_supported_correct")
        if len({e["arguments"]["query"] for e in searches}) < 2:
            result["rejections"].append("fewer_than_two_distinct_searches")
        if not links:
            result["rejections"].append("no_verified_new_entity_dependency")
        if searches:
            shadow = search.search(searches[0]["arguments"]["query"], top_k=10, mode="bm25")
            result["teacher_first_query_top10"] = coverage(gold, shadow["results"])
            if result["teacher_first_query_top10"]["all_support_documents"]:
                result["rejections"].append("teacher_first_query_top10_already_sufficient")
        result["accepted"] = not result["rejections"]
        result["rejections"] = sorted(set(result["rejections"]))
        # Partial native parallel batches are never exported as SFT.
        if result["accepted"]:
            assert result["events"][-1]["name"] == "finish"
        atomic_json(output / "episodes" / f"{rid}.json", result)
        try:
            local_request(training_url, f"/episodes/{eid}", method="DELETE")
        except (urllib.error.URLError, OSError):
            pass
    return result


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("active teacher selection\n")
    try:
        secret = load_section_key(args.key_file)
        teacher = Teacher(secret, output, args.budget_cny)
        store, search = TaskStore(args.data), SearchClient(args.search_url)
        candidates = list(rows(Path(args.data) / "multihop" / "candidates.jsonl"))
        atomic_json(output / "config.json", {"model": teacher.model, "thinking": True, "reasoning_effort": "low",
                    "max_tokens": 4096, "budget_cny": args.budget_cny, "credential_section": "sft data",
                    "target": args.target, "max_candidates": args.max_candidates,
                    "criteria": "supported correct + later query uses new observed entity + both one-shot top10 baselines miss evidence",
                    "corpus_sha256": search.meta["corpus_sha256"]})
        for candidate in candidates[:args.max_candidates]:
            summary = export(output)
            if summary["accepted"] >= args.target:
                break
            if (output / "episodes" / f"{candidate['id']}.json").exists():
                continue
            try:
                result = run_episode(candidate, teacher, store, search, output, args.training_url)
            except (BudgetStop, ProviderError) as error:
                summary = export(output)
                atomic_json(output / "stop.json", {"reason": type(error).__name__, "status": getattr(error, "status", None)})
                print(json.dumps({"stopped": type(error).__name__, "attempted": summary["attempted"], "accepted": summary["accepted"]}), flush=True)
                break
            summary = export(output)
            print(json.dumps({"id": candidate["id"], "accepted": result["accepted"], "searches": result["search_calls"],
                              "rejections": result["rejections"], "total_accepted": summary["accepted"],
                              "conservative_cny": round(teacher.usage["conservative_cny"], 4)}), flush=True)
        print(json.dumps(export(output), indent=2), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA / "multihop" / "teacher-pro-v1")
    parser.add_argument("--target", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=44)
    parser.add_argument("--budget-cny", type=float, default=5.0)
    parser.add_argument("--search-url", default="http://127.0.0.1:8020")
    parser.add_argument("--training-url", default="http://127.0.0.1:8021")
    run(parser.parse_args())
