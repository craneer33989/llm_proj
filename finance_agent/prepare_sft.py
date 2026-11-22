"""Replay first-phase failures and export verified corrections for SFT."""
from collections import Counter
from hashlib import file_digest
import json

from .agent import MODEL_ID, REVISION, assemble, model_context, parse_task
from .corpus import DB, Scope, retrieve
from .evaluate import score
from .filings import COMPANIES, DATA, RUNTIME

BASELINE = RUNTIME / "baseline-model.json"
OUTPUT = DATA / "phase2"
DIAGNOSTIC_COMPANY = "600690"


def replay_search(step, company_code, as_of):
    scope = Scope(**step["scope"])
    if scope.company_code != company_code or scope.as_of != as_of:
        raise ValueError("Recorded search does not match the question's company and cutoff")
    hits = retrieve(step["query"], scope=scope)
    if [hit["id"] for hit in hits] != [hit["id"] for hit in step["hits"]]:
        raise ValueError("Recorded evidence differs from the current index")
    return hits


def correction(original, assisted):
    case = original["case"]
    raw, fixed = original["answer"], assisted["answer"]
    prefix, failed = raw["trace"][:-1], raw["trace"][-1]
    reason = {"finish": "premature_finish", "invalid": "invalid_action"}.get(failed["action"])
    if (
        reason is None or any(step["action"] != "search" for step in prefix)
        or len(fixed["trace"]) != len(prefix) + 1 or fixed["trace"][:len(prefix)] != prefix
    ):
        raise ValueError("Expected one failed model action and a matching assisted replay")
    corrected = fixed["trace"][-1]
    if corrected["action"] != "search" or corrected["fallback_reason"] != reason:
        raise ValueError("Assisted trace did not correct the failed action")
    if corrected["model_output"] != failed["model_output"]:
        raise ValueError("Assisted trace did not reproduce the original model action")

    task, error = parse_task(case["question"])
    if error:
        raise ValueError(error)
    company_code, as_of = case["company_code"], raw["as_of"]
    company = COMPANIES[company_code][0]
    hits = [hit for step in prefix for hit in replay_search(step, company_code, as_of)]
    target, prompt = model_context(case["question"], company, task, hits)
    if (target["year"], target["period"]) != (corrected["scope"]["year"], corrected["scope"]["period"]):
        raise ValueError("Correction does not match the search scope seen by the model")
    hits.extend(replay_search(corrected, company_code, as_of))
    verified = assemble(company, task, hits)
    verified["trace"] = fixed["trace"]
    if not score(case, verified)["passed"]:
        raise ValueError("Replayed answer or citations do not match the audited question")

    split = "diagnostic" if company_code == DIAGNOSTIC_COMPANY else "train"
    sample = {
        "case_id": case["id"],
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": json.dumps({"query": corrected["query"]}, ensure_ascii=False)}],
    }
    audit = {
        "case_id": case["id"], "company_code": company_code, "group": case["group"],
        "split": split, "question": case["question"], "as_of": as_of, "reason": reason,
        "expected_status": case["status"], "expected_sources": case["sources"],
        "original_trace": raw["trace"], "replayed_trace": verified["trace"],
        "replayed_answer": verified["answer"], "replayed_citations": verified["citations"],
    }
    return split, sample, audit


def seed_records(report):
    if report.get("planner") != "model":
        raise ValueError("Use the real model baseline report")
    raw = report["runs"]["agent"]["rows"]
    assisted = {row["case"]["id"]: row for row in report["runs"]["assisted"]["rows"]}
    for original in raw:
        if original["checks"]["passed"]:
            continue
        case_id = original["case"]["id"]
        try:
            yield correction(original, assisted[case_id])
        except (ValueError, KeyError, IndexError) as error:
            raise ValueError(f"{case_id}: {error}") from error


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def prepare(output=OUTPUT):
    report = json.loads(BASELINE.read_text(encoding="utf-8"))
    records = list(seed_records(report))
    train = [sample for split, sample, _ in records if split == "train"]
    diagnostic = [sample for split, sample, _ in records if split == "diagnostic"]
    audit = [row for _, _, row in records]
    manifest = {
        "model": MODEL_ID,
        "revision": REVISION,
        "baseline_sha256": sha256(BASELINE),
        "index_sha256": sha256(DB),
        "replayed_cases": len(audit),
        "failures": dict(Counter(row["reason"] for row in audit)),
        "failures_by_case_group": dict(Counter(row["group"] for row in audit)),
        "train": {"rows": len(train), "companies": sorted({row["company_code"] for row in audit if row["split"] == "train"})},
        "diagnostic": {"rows": len(diagnostic), "company": DIAGNOSTIC_COMPANY, "origin": "一期诊断题"},
    }
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "sft-seed.jsonl", train)
    write_jsonl(output / "diagnostic-holdout.jsonl", diagnostic)
    write_jsonl(output / "failure-replays.jsonl", audit)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
