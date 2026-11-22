"""Compare the original-query baseline and two-turn agent on audited cases."""
import argparse
from collections import Counter
from decimal import Decimal
import json

from .agent import ask
from .filings import ROOT, RUNTIME


def equal_number(actual, expected):
    return actual is not None and Decimal(actual) == Decimal(expected)


def score(case, answer):
    cited = {f"{row['report_id']}:{row['page']}" for row in answer["citations"]}
    retrieved = {f"{row['report_id']}:{row['page']}" for step in answer["trace"] for row in step.get("hits", [])}
    required = set(case["sources"])
    number_correct = all(
        equal_number(answer.get("values", {}).get(field), case[field])
        for field in ("amount_cny", "yoy_percent") if field in case
    )
    status_correct = answer["status"] == case["status"]
    citations_correct = cited == required and cited <= retrieved
    segment_correct = "segment" not in case or case["segment"] == answer.get("segment")
    return {
        "passed": number_correct and status_correct and citations_correct and segment_correct,
        "number_correct": number_correct if "amount_cny" in case else None,
        "citations_correct": citations_correct,
        "both_hops_retrieved": required <= retrieved if case["group"] == "bridge" else None,
        "stopped_after_first_hop": (
            case["group"] == "bridge" and case["sources"][0] in retrieved and case["sources"][1] not in retrieved
            and any(step["action"] == "finish" for step in answer["trace"])
        ),
    }


def evaluate(planner):
    cases = [json.loads(line) for line in (ROOT / "evaluation/questions.jsonl").read_text(encoding="utf-8").splitlines()]
    runs = {}
    for mode in ("single", "agent", "assisted") if planner == "model" else ("single", "agent"):
        rows = []
        for case in cases:
            answer = ask(case["question"], case["company_code"], case.get("as_of", "2026-11-10"), mode, planner)
            rows.append({"case": case, "checks": score(case, answer), "answer": answer})
        totals = Counter(row["case"]["group"] for row in rows)
        passed = Counter(row["case"]["group"] for row in rows if row["checks"]["passed"])
        summary = {
            "cases": len(rows), "passed": sum(passed.values()),
            "by_group": {group: {"passed": passed[group], "total": total} for group, total in totals.items()},
            "numeric_correct": sum(row["checks"]["number_correct"] is True for row in rows),
            "numeric_total": sum(row["checks"]["number_correct"] is not None for row in rows),
            "citation_correct": sum(row["checks"]["citations_correct"] for row in rows),
            "both_hops_retrieved": sum(row["checks"]["both_hops_retrieved"] is True for row in rows),
            "stopped_after_first_hop": sum(row["checks"]["stopped_after_first_hop"] for row in rows),
            "average_searches": round(sum(row["answer"]["search_count"] for row in rows) / len(rows), 2),
            "evidence_fallbacks": sum(bool(step.get("fallback_reason")) for row in rows for step in row["answer"]["trace"]),
        }
        print(mode, json.dumps(summary, ensure_ascii=False), flush=True)
        runs[mode] = {"summary": summary, "rows": rows}
    (RUNTIME / f"baseline-{planner}.json").write_text(
        json.dumps({"planner": planner, "runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["model", "rules"], default="model")
    evaluate(parser.parse_args().planner)
