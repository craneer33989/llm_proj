"""Read and summarize the saved classroom comparison cohort."""
from collections import Counter
import re

from .demo_run import should_retry_with_directed_sft
from .reopd_compare import episode_table
from .scoring import sentence_key

ARMS = {
    "base-bm25": "通用 Qwen3-4B（BM25）",
    "base": "原搜索 SFT（BM25）",
    "A": "A · 定向 SFT（BM25）",
    "a-hybrid": "A · 定向 SFT（Hybrid RRF）",
    "B": "B · 教师回放 ReOPD",
    "C": "C · 非定向混合 ReOPD",
    "D": "D · 定向恢复 ReOPD",
    "route": "早答复核路由（轨迹回放）",
}
CATEGORIES = {
    "pass": "严格有证答对", "zero_wrong": "零检索答错", "zero_right": "零检索答对",
    "incomplete": "支持证据未齐", "full_wrong": "证据齐但答错",
    "format": "格式错误", "budget": "预算耗尽", "other": "其他终止",
}
COLORS = ("#13795b", "#bf4846", "#b58bd5", "#e6a343", "#52789b", "#a45676", "#698088", "#c9d1da")
CONTROLS = {"base-bm25", "a-hybrid"}
RESULT_HEADER = re.compile(r"^Doc (\d+)\(Title: [^\n]*?\) ", re.MULTILINE)


def route_source(base_episode):
    return "A" if should_retry_with_directed_sft(base_episode) else "base"


def load_tables(final, extra, expected, require_controls=False):
    tables, pending = {}, []
    for name in ARMS:
        if name == "route":
            continue
        control = name in CONTROLS
        path = (extra if control else final) / name
        if control and not require_controls and not all((path / file).is_file() for file in ("episodes.jsonl", "scores.jsonl")):
            pending.append(ARMS[name])
            continue
        table = episode_table(path)
        if table.keys() != expected or any("supported_success" not in row for row in table.values()):
            raise ValueError(f"{name} does not contain the complete scored test cohort")
        tables[name] = table
    tables["route"] = {key: tables[route_source(row)][key] for key, row in tables["base"].items()}
    return tables, pending


def search_calls(tables, arm, key):
    picked = tables[arm][key]
    if arm == "route" and route_source(tables["base"][key]) == "A":
        return tables["base"][key]["calls"] + picked["calls"]
    return picked["calls"]


def attribution(episode):
    if episode["supported_success"]:
        return "pass"
    match episode["reason"]:
        case "format_error":
            return "format"
        case "tool_budget_exhausted" | "observation_budget_exhausted":
            return "budget"
        case "finished":
            if episode["calls"] == 0:
                return "zero_right" if episode["answer_em"] else "zero_wrong"
            if episode["support_doc_coverage"] < 1 or episode["support_sentence_recall"] < 1:
                return "incomplete"
            if not episode["answer_em"]:
                return "full_wrong"
            raise ValueError(f"Unexpected strict score for episode {episode['qid']}")
        case _:
            return "other"


def failure_counts(table):
    return Counter(map(attribution, table.values()))


def support_hits(episode, facts, *, verify_score=True):
    """Match support sentences by source document within logged, model-visible snippets."""
    exposed = {}
    for step in episode["steps"]:
        if "observation" not in step:
            continue
        text, doc_ids = step["observation"], step["doc_ids"]
        headers = list(RESULT_HEADER.finditer(text))
        if [int(header[1]) for header in headers] != list(range(1, len(doc_ids) + 1)):
            return None
        for index, (header, doc_id) in enumerate(zip(headers, doc_ids, strict=True)):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
            exposed.setdefault(doc_id, []).append(sentence_key(text[header.end():end]))
    hits = [any(sentence_key(fact["text"]) in snippet for snippet in exposed.get(fact["doc_id"], ()))
            for fact in facts]
    docs = {fact["doc_id"] for fact, hit in zip(facts, hits, strict=True) if hit}
    recall = sum(hits) / len(facts) if facts else 0.0
    if verify_score and (recall != episode["support_sentence_recall"] or docs != set(episode["visible_support_docs"])):
        return None
    return hits


def answer_evidence(episode, gold, *, prefix=None):
    """Check answer-bearing labeled evidence in the logged snippets."""
    from .scoring import normalize_answer

    facts = gold["supporting_facts"]
    if prefix is not None:
        episode = {"steps": episode["steps"][:prefix]}
    hits = support_hits(episode, facts, verify_score=prefix is None)
    if hits is None:
        raise ValueError("Cannot verify the logged search observations")
    if hits and all(hits):
        return True
    aliases = [normalize_answer(answer) for answer in [gold["answer"], *gold.get("aliases", [])]]
    aliases = [answer for answer in aliases if answer and answer not in {"yes", "no", "noanswer"}]
    return any(hit and any(f" {answer} " in f" {normalize_answer(fact['text'])} " for answer in aliases)
               for hit, fact in zip(hits, facts, strict=True))


def answer_grounded(episode, gold):
    return bool(episode["answer_em"] and episode["reason"] == "finished" and answer_evidence(episode, gold))
