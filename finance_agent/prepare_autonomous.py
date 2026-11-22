"""Build a small issuer-disjoint action dataset and replay every teacher path."""
from collections import Counter
from datetime import date, timedelta
import json
from random import Random

from .agent import METRICS, assemble, parse_task, rule_next
from .autonomous import execute, messages, observation, search
from .expand_sft import candidates, fact_catalog
from .filings import COMPANIES, ROOT
from .prepare_sft import sha256, write_jsonl
from .corpus import DB

DATASET = ROOT / "data/phase2/autonomous"


def teacher_path(case):
    task, error = parse_task(case["question"])
    if error:
        raise ValueError(error)
    hits, path = [], []
    for _ in range(2 if task.source_year else 1):
        action = rule_next(task, hits)
        found = search(case, action)
        path.append(action)
        hits.extend(found)
    verified = assemble(COMPANIES[case["company_code"]][0], task, hits)
    if verified["status"] != case["status"]:
        raise ValueError(f"{case['id']}: oracle status {verified['status']} != {case['status']}")
    terminal = {"done": True, "status": case["status"]}
    return [*path, terminal]


def missing_cases(code):
    company = COMPANIES[code][0]
    for metric, name in METRICS.items():
        yield {
            "id": f"missing-{code}-future-{metric}", "company_code": code, "group": "missing", "category": "missing",
            "as_of": "2026-11-13", "question": f"{company}2026年年报的{name}和同比是多少？",
            "status": "insufficient", "sources": [], "missing_reason": "report_outside_library",
        }
        report = next(row for row in fact_catalog() if row["company_code"] == code and row["year"] == 2025 and row["period"] == "H1")
        cutoff = (date.fromisoformat(report["published_at"]) - timedelta(days=1)).isoformat()
        yield {
            "id": f"missing-{code}-cutoff-{metric}", "company_code": code, "group": "missing", "category": "missing",
            "as_of": cutoff, "question": f"{company}2025年半年报的{name}和同比是多少？",
            "status": "insufficient", "sources": [], "missing_reason": "not_disclosed_by_cutoff",
        }
    source = next(row for row in fact_catalog() if row["company_code"] == code and row["year"] == 2024
                  and row["period"] == "FY" and row["fact"]["kind"] == "segments")
    for rank, word in enumerate(("最高", "第二高", "第三高"), 1):
        yield {
            "id": f"missing-{code}-bridge-{rank}", "company_code": code, "group": "missing", "category": "missing",
            "as_of": "2026-11-13", "question": f"{company}2024年年报中收入{word}的板块，在2026年半年报的营业收入和同比是多少？",
            "status": "insufficient", "sources": [f"{source['report_id']}:{source['page']}"],
            "missing_reason": "second_report_outside_library",
        }


def prepare(output=DATASET):
    all_cases = list(candidates(fact_catalog()))
    rng = Random(42)
    train, evaluation = [], []
    for code in COMPANIES:
        direct = [dict(row, category="direct") for row in all_cases if row["company_code"] == code and row["group"] == "direct"]
        bridges = [dict(row, category="bridge") for row in all_cases if row["company_code"] == code and row["group"] == "bridge"]
        rng.shuffle(bridges)
        if code == "600690":
            evaluation.extend(direct[:10] + bridges[:10])
            for case in direct[:6] + bridges[:4]:
                evaluation.append(dict(case, id=f"stop-{case['id']}", category="stop",
                                       initial_searches=teacher_path(case)[:-1]))
            missing = list(missing_cases(code))
            # Ten distinct missing-evidence tasks, including a different target report.
            extra = dict(missing[-1], id="missing-600690-bridge-next-year", question=missing[-1]["question"].replace("2026年半年报", "2026年年报"))
            evaluation.extend(missing + [extra])
        else:
            train.extend(direct + bridges[:6] + list(missing_cases(code)))
    actions, replayed = [], []
    for case in train:
        observations = []
        path = teacher_path(case)
        verified = execute(case, lambda _: json.dumps(path.pop(0), ensure_ascii=False))
        if not verified["checks"]["passed"] or path:
            raise ValueError(f"{case['id']}: teacher rollout failed validation")
        for turn, action in enumerate(teacher_path(case), 1):
            actions.append({
                "case_id": case["id"], "company_code": case["company_code"], "category": case["category"],
                "turn": turn, "action_type": "stop" if action.get("done") else "search",
                "prompt": messages(case, observations),
                "completion": [{"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}],
            })
            if "query" in action:
                found = search(case, action)
                observations.append(observation(action, found))
        replayed.append(verified)
    for case in evaluation:
        path = [teacher_path(case)[-1]] if case["category"] == "stop" else teacher_path(case)
        rollout = execute(case, lambda _: json.dumps(path.pop(0), ensure_ascii=False))
        if not rollout["checks"]["passed"] or path:
            raise ValueError(f"{case['id']}: held-out teacher replay failed")
        replayed.append(rollout)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train-actions.jsonl", actions)
    write_jsonl(output / "train-episodes.jsonl", train)
    write_jsonl(output / "eval-episodes.jsonl", evaluation)
    write_jsonl(output / "teacher-replays.jsonl", replayed)
    manifest = {
        "train_episodes": len(train), "train_actions": len(actions), "eval_episodes": len(evaluation),
        "total_records": len(actions) + len(evaluation),
        "train_categories": dict(Counter(case["category"] for case in train)),
        "train_action_types": dict(Counter(row["action_type"] for row in actions)),
        "eval_categories": dict(Counter(case["category"] for case in evaluation)),
        "train_companies": ["000333", "000651"], "eval_companies": ["600690"],
        "all_teacher_replays_passed": True, "corpus_sha256": sha256(DB),
        "next_step_hints": False, "search_scope_chosen_by_model": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
