"""Generate new audited questions and replay their search trajectories."""
from collections import Counter
from decimal import Decimal
import json
import sqlite3

from .agent import METRICS, RANK_WORDS, ask, model_context, parse_task, period_name, value_in_yuan
from .corpus import DB
from .evaluate import score
from .filings import ROOT
from .prepare_sft import DIAGNOSTIC_COMPANY, OUTPUT, replay_search, sha256, write_jsonl

EXPANDED = OUTPUT / "expanded"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fact_catalog():
    con = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in con.execute("SELECT * FROM evidence WHERE fact IS NOT NULL ORDER BY id")]
    finally:
        con.close()
    for row in rows:
        row["fact"] = json.loads(row["fact"])
    return rows


def task_key(case):
    task, _ = parse_task(case["question"])
    if task:
        return case["company_code"], task.year, task.period, task.metric, task.source_year, task.rank


def expected_values(fact):
    # The oracle reads the disclosed percentage, independently of the
    # runtime calculator, which normally recomputes it from the two columns.
    return {
        "amount_cny": str(value_in_yuan(fact["current"], fact["unit"])),
        "yoy_percent": str(Decimal(fact["yoy"])),
    }


def source_key(row):
    return f"{row['report_id']}:{row['page']}"


def candidates(rows):
    segments = [row for row in rows if row["fact"]["kind"] == "segments"]
    for row in rows:
        if row["fact"]["kind"] != "metric":
            continue
        fact = row["fact"]
        yield {
            "id": f"x-{row['report_id']}-{fact['metric']}",
            "group": "direct", "company_code": row["company_code"], "as_of": row["published_at"],
            "question": f"{period_name(row['year'], row['period'])}披露的{METRICS[fact['metric']]}及同比增减是多少？",
            "status": "answered", "sources": [source_key(row)], **expected_values(fact),
        }
    for source in segments:
        if source["period"] != "FY":
            continue
        for target in segments:
            if target["company_code"] != source["company_code"] or target["report_id"] == source["report_id"]:
                continue
            target_products = {product["label"]: product for product in target["fact"]["products"]}
            for rank, product in enumerate(source["fact"]["products"][:len(RANK_WORDS)], 1):
                label = product["label"]
                if label not in target_products:
                    continue
                yield {
                    "id": f"x-{source['report_id']}-to-{target['year']}-{target['period']}-r{rank}",
                    "group": "bridge", "company_code": source["company_code"],
                    "as_of": max(source["published_at"], target["published_at"]),
                    "question": f"{source['year']}年年报分产品收入{RANK_WORDS[rank - 1]}的板块，在"
                                f"{period_name(target['year'], target['period'])}的营业收入和同比是多少？",
                    "status": "answered", "segment": label,
                    "sources": [source_key(source), source_key(target)], **expected_values(target_products[label]),
                }


def replay_case(case):
    question, code, as_of = (case[key] for key in ("question", "company_code", "as_of"))
    replay = ask(question, code, as_of, planner="rules")
    checks = score(case, replay)
    if not checks["passed"]:
        raise ValueError(f"{case['id']}: replay failed oracle checks {checks}")
    task, _ = parse_task(question)
    hits, actions = [], []
    for turn, step in enumerate(replay["trace"], 1):
        if step["action"] != "search":
            raise ValueError(f"{case['id']}: successful replay contains a non-search step")
        target, prompt = model_context(question, replay["citations"][0]["company"], task, hits)
        if (target["year"], target["period"]) != (step["scope"]["year"], step["scope"]["period"]):
            raise ValueError(f"{case['id']}: training scope differs from replay")
        actions.append({
            "case_id": case["id"], "turn": turn, "group": case["group"], "prompt": prompt,
            "completion": [{"role": "assistant", "content": json.dumps({"query": step["query"]}, ensure_ascii=False)}],
        })
        hits.extend(replay_search(step, code, as_of))
    expected_turns = 2 if case["group"] == "bridge" else 1
    if len(actions) != expected_turns:
        raise ValueError(f"{case['id']}: replay used {len(actions)} searches, expected {expected_turns}")
    replay.pop("latency_ms")
    return actions, {"case": case, "checks": checks, "replay": replay}


def expand(output=EXPANDED):
    seen = {task_key(case) for case in read_jsonl(ROOT / "evaluation/questions.jsonl")}
    train_cases, diagnostic, actions, audit = [], [], [], []
    for case in candidates(fact_catalog()):
        key = task_key(case)
        if key is None:
            raise ValueError(f"{case['id']}: unsupported generated question")
        if key in seen:
            continue
        seen.add(key)
        steps, record = replay_case(case)
        if case["company_code"] == DIAGNOSTIC_COMPANY:
            diagnostic.append(case)
            record["split"] = "diagnostic"
        else:
            train_cases.append(case)
            actions.extend(steps)
            record["split"] = "train"
        audit.append(record)

    if not train_cases or not diagnostic:
        raise ValueError("Expansion requires training and held-out-company questions")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train-questions.jsonl", train_cases)
    write_jsonl(output / "train-actions.jsonl", actions)
    write_jsonl(output / "diagnostic-questions.jsonl", diagnostic)
    write_jsonl(output / "verified-replays.jsonl", audit)
    manifest = {
        "train_questions": dict(Counter(case["group"] for case in train_cases)),
        "train_actions": len(actions),
        "diagnostic_company": DIAGNOSTIC_COMPANY,
        "diagnostic_questions": dict(Counter(case["group"] for case in diagnostic)),
        "replayed_questions": len(audit),
        "corpus_sha256": sha256(DB),
        "phase1_questions_sha256": sha256(ROOT / "evaluation/questions.jsonl"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(expand(), ensure_ascii=False, indent=2))
