"""Search-R1 protocol and state replay on the local HotpotQA environment."""
from dataclasses import asdict
import hashlib
import json
import re
import sqlite3

from .download import sha256
from .scoring import answer_metrics, sentence_key
from .training import EpisodeConfig, TrainingEpisode

UPSTREAM = "7a499d29ab07ff50dfde039f9d7a44dfe313b95e"
CONFIG = EpisodeConfig()
SYSTEM = r'''You are a helpful assistant that can use tools to answer questions. You have access to a search tool. When you need to look up information, call the search tool as described below.

# Tools

You may call one function at a time to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
<tool name="search" description="Search for information." args="{\"query\": string}" />
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
After a tool is executed, you will receive the tool result in a user message wrapped in <tool_response></tool_response> tags.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>'''
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}\s*\Z")
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>\s*\Z", re.DOTALL)
RESERVED = re.compile(r"<\|[^<>\n]+\|>|</?tool_response>")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def clean(text):
    return RESERVED.sub(lambda match: match[0].replace("<", "＜").replace(">", "＞"), text)


def initial_messages(question):
    user = (r"Work through the question step by step. Use the search tool when you need more information. "
            r"Put the short final answer on a separate last line as Answer: \boxed{your answer}."
            f"\n\nQuestion: {clean(question)}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def render(messages):
    return "".join(
        f"<|im_start|>user\n<tool_response>\n{msg['content']}\n</tool_response><|im_end|>\n"
        if msg["role"] == "tool" else f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        for msg in messages
    ) + "<|im_start|>assistant\n"


def parse(text):
    if RESERVED.search(text):
        return "format_error", None
    call = CALL.search(text)
    if call and text.count("<tool_call>") == text.count("</tool_call>") == 1:
        try:
            payload = json.loads(call[1])
            if isinstance(payload, dict) and set(payload) == {"name", "arguments"} and payload["name"] == "search":
                args = payload["arguments"]
                if isinstance(args, dict) and set(args) == {"query"}:
                    query = args["query"]
                    if isinstance(query, str) and 1 <= len(query.strip()) <= 512:
                        return "search", query.strip()
        except ValueError:
            pass
    if "<tool_call" not in text and "</tool_call" not in text:
        answer = BOXED.search(text)
        if answer and 1 <= len(answer[1].strip()) <= 2000:
            return "finish", answer[1].strip()
    return "format_error", None


def observation_text(observation):
    return "\n".join(f"Doc {number}(Title: {clean(hit['title'])}) {clean(hit['text'])}"
                     for number, hit in enumerate(observation.get("results", []), 1)) or "No search results."


class BackendError(RuntimeError):
    pass


class Rollout:
    def __init__(self, engine, task, split, prefix=()):
        self.env = TrainingEpisode(engine, task, split, CONFIG)
        self.messages = initial_messages(task["question"])
        self.steps, self.exposed = [], {}
        self.reason = None
        for step in prefix:
            self.advance(step["raw"])
            if self.reason or self.steps[-1] != step:
                raise ValueError("Prefix replay mismatch")

    def advance(self, raw):
        if self.reason:
            raise ValueError("Episode is already terminal")
        kind, value = parse(raw)
        step = {"raw": raw, "kind": kind}
        self.steps.append(step)
        self.messages.append({"role": "assistant", "content": raw})
        if kind == "format_error":
            self.reason = kind
            return
        try:
            result = self.env.step("local_search", {"query": value, "top_k": CONFIG.max_top_k}) if kind == "search" else (
                self.env.step("finish", {"answer": value, "citations": []}))
        except (sqlite3.Error, OSError) as error:
            raise BackendError(str(error)) from error
        observation = result["observation"]
        if observation.get("retryable"):
            raise BackendError(observation["error"])
        step["value"] = value
        step["observation_hash"] = digest(observation)
        if kind == "finish":
            self.reason = "finished"
        elif result["truncated"] or "error" in observation:
            self.reason = observation.get("error", "truncated")
        else:
            step["observation"] = observation_text(observation)
            step["doc_ids"] = [hit["doc_id"] for hit in observation["results"]]
            for hit in observation["results"]:
                self.exposed.setdefault(hit["doc_id"], []).append(clean(hit["text"]))
            self.messages.append({"role": "tool", "content": step["observation"]})

    def record(self, gold, attempt):
        facts = gold["supporting_facts"]
        expected_docs = {fact["doc_id"] for fact in facts}
        coverage = len(expected_docs & self.exposed.keys()) / len(expected_docs) if expected_docs else 0.0
        visible = [fact for fact in facts if any(sentence_key(fact["text"]) in sentence_key(text)
                                               for text in self.exposed.get(fact["doc_id"], []))]
        em, f1 = max(answer_metrics(self.env.answer, reference)
                     for reference in [gold["answer"], *gold.get("aliases", [])])
        supported = bool(self.reason == "finished" and em and coverage == 1 and len(visible) == len(facts))
        qid = self.env.task["id"]
        return ({"qid": qid, "split": self.env.split, "attempt": attempt, "messages": self.messages,
                 "steps": self.steps, "reason": self.reason, "calls": len(self.env.session.trace)},
                {"qid": qid, "attempt": attempt, "answer_em": em, "answer_f1": f1,
                 "support_doc_coverage": coverage, "support_sentence_recall": len(visible) / len(facts) if facts else 0,
                 "visible_support_docs": sorted({fact["doc_id"] for fact in visible}),
                 "answer_with_all_support_docs": bool(em and coverage == 1), "supported_success": supported,
                 "early_stop": (self.reason == "finished" and [step["kind"] for step in self.steps] ==
                                ["search", "finish"] and len(visible) < len(facts))})


def environment_lock(engine):
    return {"upstream": UPSTREAM, "config": asdict(CONFIG), "protocol_sha256": digest(render(initial_messages(""))),
            "split_manifest_sha256": sha256(engine.data / "manifest.json"),
            "index_manifest_sha256": sha256(engine.index / "manifest.json"),
            "index_sha256": sha256(engine.index / "documents.sqlite3")}
